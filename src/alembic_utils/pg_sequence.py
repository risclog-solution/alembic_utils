import re
from typing import Optional

from sqlalchemy import DefaultClause
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from alembic_utils.replaceable_entity import ReplaceableEntity


class PGSequence(ReplaceableEntity):
    """
    ReplaceableEntity for PostgreSQL SEQUENCE objects.

    **Parameters:**

    * **schema** (*str*): The database schema, e.g., 'public'
    * **signature** (*str*): The unique name of the sequence within the schema
    * **definition** (*str*): The SQL definition of the sequence as it exists in the database

    **Example:**

        seq = PGSequence(
            schema="public",
            signature="my_sequence",
            definition=\"\"\"
                AS bigint
                START WITH 1
                INCREMENT BY 1
                MINVALUE 1
                MAXVALUE 9223372036854775807
                CACHE 1
                NO CYCLE
            \"\"\"
        )
    """

    _create_counter = 0

    def __init__(
        self,
        schema: str,
        signature: str,
        definition: str,
    ):
        self.schema = schema
        self.signature = signature
        clean_def = self.clean_sequence_definition(definition)
        self.definition = clean_def

        self.data_type = self._extract(r"AS\s+(\w+)", clean_def, default="bigint")
        self.start_value = int(self._extract(r"START WITH\s+(-?\d+)", clean_def, default=1))
        self.increment = int(self._extract(r"INCREMENT BY\s+(-?\d+)", clean_def, default=1))
        self.minvalue = int(self._extract(r"MINVALUE\s+(-?\d+)", clean_def, default=1))
        self.maxvalue = int(
            self._extract(r"MAXVALUE\s+(-?\d+)", clean_def, default=9223372036854775807)
        )
        self.cache = int(self._extract(r"CACHE\s+(\d+)", clean_def, default=1))
        self.cycle = bool(re.search(r"\bCYCLE\b", clean_def, re.IGNORECASE)) and not re.search(
            r"NO CYCLE", clean_def, re.IGNORECASE
        )
        owned_match = re.search(r"OWNED BY\s+([a-zA-Z0-9_.\"]+)", clean_def, re.IGNORECASE)
        self.owned_by = owned_match.group(1) if owned_match else None

        super().__init__(self.schema, self.signature, self.definition)

    def _extract(self, regex, source, default):
        match = re.search(regex, source, re.IGNORECASE)
        return match.group(1) if match else default

    @property
    def type_(self) -> str:
        return "sequence"

    @staticmethod
    def clean_sequence_definition(definition: str) -> str:
        definition = re.sub(r"--.*", "", definition)
        lines = [line.strip() for line in definition.splitlines() if line.strip()]
        definition = " ".join(lines)
        definition = re.sub(r"\s+", " ", definition)
        definition = re.sub(r"\s*,\s*", ", ", definition)
        return definition.strip()

    @classmethod
    def from_args(
        cls,
        schema: str,
        signature: str,
        data_type: str = "bigint",
        start_value: int = 1,
        minvalue: int = 1,
        maxvalue: int = 9223372036854775807,
        increment: int = 1,
        cycle: bool = False,
        cache: int = 1,
        owned_by: Optional[str] = None,
    ):
        """
        Ergonomic factory to build a sequence from Python arguments.
        """
        lines = [
            f"AS {data_type}",
            f"START WITH {start_value}",
            f"INCREMENT BY {increment}",
            f"MINVALUE {minvalue}",
            f"MAXVALUE {maxvalue}",
            f"CACHE {cache}",
            "CYCLE" if cycle else "NO CYCLE",
        ]
        if owned_by:
            lines.append(f"OWNED BY {owned_by}")
        definition = "\n    " + "\n    ".join(lines)
        definition = PGSequence.clean_sequence_definition(definition)
        return cls(schema, signature, definition)

    @classmethod
    def from_database(cls, sess: Session, schema: str = "%"):
        sql = """
        SELECT c.relname as sequence_name,
               n.nspname as sequence_schema
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind = 'S'
          AND n.nspname LIKE :schema
        """

        rows = list(sess.execute(sql_text(sql), {"schema": schema}).mappings())

        entities = []
        for row in rows:

            sql2 = """
            SELECT s.seqtypid::regtype::text as data_type,
                   s.seqstart as start_value,
                   s.seqmin as minvalue,
                   s.seqmax as maxvalue,
                   s.seqincrement as increment,
                   s.seqcycle as cycle,
                   s.seqcache as cache
            FROM pg_sequence s
            JOIN pg_class c ON c.oid = s.seqrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = :schema AND c.relname = :sequence
            """
            result = (
                sess.execute(
                    sql_text(sql2),
                    {
                        "schema": row["sequence_schema"],
                        "sequence": row["sequence_name"],
                    },
                )
                .mappings()
                .fetchone()
            )
            if not result:
                sql3 = """
                SELECT data_type,
                       start_value::bigint,
                       minimum_value::bigint as minvalue,
                       maximum_value::bigint as maxvalue,
                       increment::bigint,
                       cycle_option
                FROM information_schema.sequences
                WHERE sequence_schema = :schema AND sequence_name = :sequence
                """
                result2 = (
                    sess.execute(
                        sql_text(sql3),
                        {
                            "schema": row["sequence_schema"],
                            "sequence": row["sequence_name"],
                        },
                    )
                    .mappings()
                    .fetchone()
                )
                if not result2:
                    continue
                cycle = result2["cycle_option"] == "YES"
                cache = 1
                data_type = result2["data_type"]
                start_value = result2["start_value"]
                minvalue = result2["minvalue"]
                maxvalue = result2["maxvalue"]
                increment = result2["increment"]
            else:
                cycle = result["cycle"]
                cache = result["cache"]
                data_type = result["data_type"]
                start_value = result["start_value"]
                minvalue = result["minvalue"]
                maxvalue = result["maxvalue"]
                increment = result["increment"]

            lines = [
                f"AS {data_type}",
                f"START WITH {start_value}",
                f"INCREMENT BY {increment}",
                f"MINVALUE {minvalue}",
                f"MAXVALUE {maxvalue}",
                f"CACHE {cache or 1}",
                "CYCLE" if cycle else "NO CYCLE",
            ]
            definition = "\n    " + "\n    ".join(lines)
            definition = PGSequence.clean_sequence_definition(definition)
            entities.append(
                cls(
                    schema=row["sequence_schema"],
                    signature=row["sequence_name"],
                    definition=definition,
                )
            )

        return entities

    def to_sql_statement_create(self):
        self._create_counter += 1
        sql = f"CREATE SEQUENCE {self.schema}.{self.signature} {self.definition};"

        return sql_text(sql)

    def to_sql_statement_drop(self, cascade=False):
        sql = f"DROP SEQUENCE IF EXISTS {self.schema}.{self.signature} CASCADE"
        if cascade:
            sql += " CASCADE"
        sql += ";"

        return sql_text(sql)

    def to_sql_statement_create_or_replace(self):
        yield self.to_sql_statement_drop()
        yield self.to_sql_statement_create()

    def __repr__(self):
        return (
            f"PGSequence(schema={self.schema!r}, signature={self.signature!r}, "
            f"definition={self.definition!r})"
        )

    @classmethod
    def render_import_statement(cls) -> str:
        module_path = cls.__module__
        class_name = cls.__name__
        return f"from {module_path} import {class_name}"

    def next_value(self):
        full_name = f"{self.schema}.{self.signature}" if self.schema else self.signature

        return DefaultClause(sql_text(f"nextval('{full_name}')"))
