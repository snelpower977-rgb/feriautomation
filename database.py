from __future__ import annotations

import uuid
from typing import Any

import mysql.connector
from mysql.connector import pooling

from config import settings


def _connection_kwargs() -> dict[str, Any]:
    return dict(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        autocommit=False,
    )


class DatabaseClient:
    """MySQL access.

    - ``use_pool=True`` (DB writer): small connection pool pre-opened at init.
    - ``use_pool=False`` (workers, CLI): **one** persistent connection per process.
      Using a pool per worker with ``pool_size=10`` exhausts ``max_connections`` fast.
    """

    def __init__(self, *, use_pool: bool = False) -> None:
        self._use_pool = use_pool
        self._pool: pooling.MySQLConnectionPool | None = None
        self._single: mysql.connector.MySQLConnection | None = None
        if use_pool:
            self._pool = pooling.MySQLConnectionPool(
                pool_name=settings.mysql_pool_name,
                pool_size=max(1, settings.mysql_pool_size),
                pool_reset_session=True,
                **_connection_kwargs(),
            )

    def _borrow(self) -> mysql.connector.MySQLConnection:
        if self._use_pool and self._pool is not None:
            return self._pool.get_connection()
        if self._single is None or not self._single.is_connected():
            self._single = mysql.connector.connect(**_connection_kwargs())
        else:
            try:
                self._single.ping(reconnect=True, attempts=3, delay=1)
            except mysql.connector.Error:
                try:
                    self._single.close()
                except mysql.connector.Error:
                    pass
                self._single = mysql.connector.connect(**_connection_kwargs())
        return self._single

    def close(self) -> None:
        if self._single is not None:
            try:
                self._single.close()
            except mysql.connector.Error:
                pass
            self._single = None
        self._pool = None

    def _checkout(self) -> mysql.connector.MySQLConnection:
        if self._use_pool and self._pool is not None:
            return self._pool.get_connection()
        return self._borrow()

    def _release(self, conn: mysql.connector.MySQLConnection) -> None:
        if self._use_pool:
            try:
                conn.close()
            except mysql.connector.Error:
                pass

    def ensure_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS bl_documents (
            id CHAR(36) PRIMARY KEY,
            file_name VARCHAR(512) NOT NULL,
            file_hash CHAR(64) NOT NULL,
            bl_number VARCHAR(128),
            booking_number VARCHAR(128),
            vessel VARCHAR(255),
            port_loading VARCHAR(255),
            port_discharge VARCHAR(255),
            weight VARCHAR(128),
            shipper TEXT,
            consignee TEXT,
            raw_text LONGTEXT,
            status ENUM('pending', 'processed', 'failed') NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_file_hash (file_hash),
            INDEX idx_bl_number (bl_number),
            INDEX idx_created_at (created_at),
            INDEX idx_status (status)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        conn = self._checkout()
        try:
            with conn.cursor() as cursor:
                cursor.execute(ddl)
            conn.commit()
        finally:
            self._release(conn)
        self._migrate_schema_add_columns(
            [
                ("shipper", "TEXT NULL"),
                ("consignee", "TEXT NULL"),
            ]
        )
        self._ensure_audit_view()

    def _migrate_schema_add_columns(self, columns: list[tuple[str, str]]) -> None:
        conn = self._checkout()
        try:
            with conn.cursor() as cursor:
                for name, coltype in columns:
                    try:
                        cursor.execute(
                            f"ALTER TABLE bl_documents ADD COLUMN {name} {coltype}"
                        )
                        conn.commit()
                    except mysql.connector.Error as exc:
                        if exc.errno != 1060:
                            raise
                        conn.rollback()
        finally:
            self._release(conn)

    def _ensure_audit_view(self) -> None:
        ddl = """
        CREATE OR REPLACE VIEW v_bl_documents_audit AS
        SELECT
            id,
            file_name,
            file_hash,
            status,
            created_at,
            bl_number,
            booking_number,
            CASE WHEN file_hash = 'missing' THEN 1 ELSE 0 END AS is_missing_hash,
            CASE
                WHEN COALESCE(NULLIF(TRIM(bl_number), ''), NULLIF(TRIM(booking_number), '')) IS NULL
                THEN 1 ELSE 0
            END AS is_identity_missing
        FROM bl_documents
        """
        conn = self._checkout()
        try:
            with conn.cursor() as cursor:
                cursor.execute(ddl)
            conn.commit()
        finally:
            self._release(conn)

    def file_hash_exists(self, file_hash: str) -> bool:
        # Deduplicate only against successfully processed documents.
        # Failed/partial attempts must not cause future files to be skipped.
        query = "SELECT 1 FROM bl_documents WHERE file_hash = %s AND status = 'processed' LIMIT 1"
        conn = self._checkout()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, (file_hash,))
                row = cursor.fetchone()
            return bool(row)
        finally:
            self._release(conn)

    def insert_batch(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        query = """
        INSERT INTO bl_documents (
            id, file_name, file_hash, bl_number, booking_number, vessel,
            port_loading, port_discharge, weight, shipper, consignee,
            raw_text, status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            bl_number = VALUES(bl_number),
            booking_number = VALUES(booking_number),
            vessel = VALUES(vessel),
            port_loading = VALUES(port_loading),
            port_discharge = VALUES(port_discharge),
            weight = VALUES(weight),
            shipper = VALUES(shipper),
            consignee = VALUES(consignee),
            raw_text = VALUES(raw_text),
            status = VALUES(status)
        """
        params = [
            (
                record.get("id", str(uuid.uuid4())),
                record["file_name"],
                record["file_hash"],
                record.get("bl_number"),
                record.get("booking_number"),
                record.get("vessel"),
                record.get("port_loading"),
                record.get("port_discharge"),
                record.get("weight"),
                record.get("shipper"),
                record.get("consignee"),
                record.get("raw_text"),
                record.get("status", "processed"),
            )
            for record in records
        ]
        conn = self._checkout()
        try:
            with conn.cursor() as cursor:
                cursor.executemany(query, params)
            conn.commit()
        finally:
            self._release(conn)
        return len(params)

    def count_recent_processed(self, minutes: int = 1) -> int:
        query = """
        SELECT COUNT(*) FROM bl_documents
        WHERE status = 'processed'
          AND created_at >= (NOW() - INTERVAL %s MINUTE)
        """
        conn = self._checkout()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, (minutes,))
                row = cursor.fetchone()
            return int(row[0] if row else 0)
        finally:
            self._release(conn)

    def count_processed_total(self) -> int:
        query = "SELECT COUNT(*) FROM bl_documents WHERE status = 'processed'"
        conn = self._checkout()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query)
                row = cursor.fetchone()
            return int(row[0] if row else 0)
        finally:
            self._release(conn)

    def list_recent_file_names(self, minutes: int = 120) -> list[str]:
        query = """
        SELECT DISTINCT file_name
        FROM bl_documents
        WHERE created_at >= (NOW() - INTERVAL %s MINUTE)
        """
        conn = self._checkout()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, (minutes,))
                rows = cursor.fetchall() or []
            return [str(r[0]) for r in rows if r and r[0]]
        finally:
            self._release(conn)

    def count_recent_missing_hash(self, minutes: int = 120) -> int:
        query = """
        SELECT COUNT(*)
        FROM v_bl_documents_audit
        WHERE created_at >= (NOW() - INTERVAL %s MINUTE)
          AND is_missing_hash = 1
        """
        conn = self._checkout()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, (minutes,))
                row = cursor.fetchone()
            return int(row[0] if row else 0)
        finally:
            self._release(conn)
