# pylint: disable=unused-argument,invalid-name,line-too-long
import re
from typing import Any, Generator, List, Optional

from parse import parse
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import TextClause

from alembic_utils.exceptions import SQLParseFailure
from alembic_utils.replaceable_entity import ReplaceableEntity
from alembic_utils.statement import normalize_whitespace

__all__ = [
    "PGAggregate",
    "pg_aggregate_from_definition",
    "SQLParseFailure",
    "ReplaceableEntity",
    "normalize_whitespace",
]

# PG built-in function lists, separated by type for easy lookup
# fmt: off
PG_CATALOG_FUNCTIONS_NUMERIC = {
    # Numeric/Math
    "abs", "acos", "acosd", "acosh", "asin", "asind", "asinh", "atan", "atan2", "atan2d", "atand", "atanh",
    "cbrt", "ceil", "ceiling", "degrees", "div", "exp", "floor", "ln", "log", "log10", "mod", "pi", "power",
    "radians", "round", "sign", "sin", "sind", "sinh", "sqrt", "tan", "tand", "tanh", "trunc", "random",
    "int2pl", "int4pl", "int8pl", "float4pl", "float8pl", "numeric_add",
    "int2mul", "int4mul", "int8mul", "float4mul", "float8mul", "numeric_mul",
    "int2sum", "int4sum", "int8sum", "float4sum", "float8sum", "numeric_sum",
    "int2_accum", "int4_accum", "int8_accum", "float4_accum", "float8_accum", "numeric_accum",
    "int2_avg_accum", "int4_avg_accum", "int8_avg_accum",
    "avg", "count", "sum", "stddev_pop", "stddev_samp", "var_pop", "var_samp", "variance",
    "min", "max", "greatest", "least",
}

PG_CATALOG_FUNCTIONS_STRING = {
    # String/Text
    "ascii", "bpcharcat", "chr", "concat", "concat_ws", "format", "initcap", "left", "length", "lower",
    "lpad", "ltrim", "md5", "position", "regexp_matches", "regexp_replace", "repeat", "replace", "reverse",
    "right", "rpad", "rtrim", "split_part", "strpos", "substr", "substring", "to_ascii", "to_hex", "trim", "upper",
    "textcat", "string_agg", "array_to_string", "array_agg", "btrim",
}

PG_CATALOG_FUNCTIONS_DATETIME = {
    # Date/Time
    "age", "current_date", "current_time", "current_timestamp", "date_part", "date_trunc", "extract", "isfinite",
    "localtimestamp", "now", "statement_timestamp", "timeofday", "transaction_timestamp", "to_char",
    "to_date", "to_timestamp", "interval", "make_date", "make_interval", "make_time", "make_timestamp", "make_timestamptz",
}

PG_CATALOG_FUNCTIONS_ARRAY = {
    # Array
    "array_agg", "array_append", "array_cat", "array_dims", "array_length", "array_lower", "array_ndims",
    "array_position", "array_positions", "array_prepend", "array_remove", "array_replace", "array_to_string",
    "array_upper", "unnest", "array_fill",
}

PG_CATALOG_FUNCTIONS_BOOL_BIT = {
    # Bool/Bitwise
    "bool_and", "bool_or", "booland_statefunc", "boolor_statefunc", "bool_accum", "bool_accum_inv", "bool_alltrue", "bool_anytrue",
    "bit_and", "bit_or", "bit_xor", "bit_length", "get_bit", "get_byte", "set_bit", "set_byte",
}

PG_CATALOG_FUNCTIONS_JSON = {
    # JSON
    "json_agg", "json_agg_finalfn", "json_agg_transfn", "json_array_elements", "json_array_elements_text",
    "json_object_agg", "json_object_agg_finalfn", "json_object_agg_transfn", "json_object_keys", "to_json", "to_jsonb",
    "row_to_json", "array_to_json", "json_build_array", "json_build_object", "jsonb_agg", "jsonb_object_agg",
    "jsonb_object_keys", "jsonb_array_elements", "jsonb_array_elements_text", "jsonb_set",
}

PG_CATALOG_FUNCTIONS_STATS = {
    # Statistical
    "corr", "covar_pop", "covar_samp", "regr_avgx", "regr_avgy", "regr_count", "regr_intercept", "regr_r2",
    "regr_slope", "regr_sxx", "regr_sxy", "regr_syy", "mode", "percentile_cont", "percentile_disc", "rank",
    "dense_rank", "cume_dist", "ntile", "row_number", "first_value", "last_value", "nth_value",
}

PG_CATALOG_FUNCTIONS_GEOMETRY = {
    # Geometry/Network
    "box", "circle", "lseg", "path", "point", "polygon", "area", "center", "diameter", "height",
    "length", "npoints", "radius", "width", "host", "hostmask", "netmask", "broadcast", "network", "text",
}

PG_CATALOG_FUNCTIONS_MISC = {
    # Misc/System/Utility
    "coalesce", "nullif", "pg_typeof", "version", "quote_ident", "quote_literal", "quote_nullable",
    "current_database", "current_schema", "current_user", "nextval", "currval", "setval",
    "encode", "decode", "digest", "hmac", "pg_backend_pid", "pg_postmaster_start_time",
    "to_number", "cast",
}
# fmt: on


class PGAggregate(ReplaceableEntity):
    """
    PGAggregate allows the creation and management of PostgreSQL
    aggregate functions compatible with Alembic migrations.

    **Parameters:**
        * **schema* str: Schema name where the aggregate function is defined.
        * **signature* str: Function signature, e.g., 'agg_func(integer)'.
        * **definition* str: Aggregate definition, e.g., 'SFUNC = func_name, STYPE = integer'.
        * **_sfunc* str: Name of the state transition function.
        * **_stype* str: Data type of the state value.
        * **_initcond* Any: Initial condition of the aggregate.
        * **_finalfunc* str: Optional final function applied at aggregation end.
        * **_stored_dependencies* list: Explicit dependency tracking.
        * **all_entities* list: List of entities for automatic dependency resolution.
        * **kwargs* Any: Additional PostgreSQL-specific options (e.g., PARALLEL).
    """

    type_ = "aggregate"
    KNOWN_INT_TYPES = {"int", "int2", "int4", "int8", "bigint", "smallint", "integer"}
    PG_CATALOG_FUNCTIONS = (
        PG_CATALOG_FUNCTIONS_NUMERIC
        | PG_CATALOG_FUNCTIONS_STRING
        | PG_CATALOG_FUNCTIONS_DATETIME
        | PG_CATALOG_FUNCTIONS_ARRAY
        | PG_CATALOG_FUNCTIONS_BOOL_BIT
        | PG_CATALOG_FUNCTIONS_JSON
        | PG_CATALOG_FUNCTIONS_STATS
        | PG_CATALOG_FUNCTIONS_GEOMETRY
        | PG_CATALOG_FUNCTIONS_MISC
    )

    def __init__(
        self,
        schema: str,
        signature: str,
        definition: str,
        _sfunc: Optional[str] = None,
        _stype: Optional[str] = None,
        _initcond: Optional[Any] = None,
        _finalfunc: Optional[str] = None,
        _stored_dependencies: Optional[List[Any]] = None,
        all_entities: Optional[List[Any]] = None,
        **kwargs: Any,
    ):
        # 1. Automatically detect INITCOND if not given, for known types
        if _initcond is None and _stype is not None:
            _initcond = self.autofill_initcond_for_type(_stype)

        # 2. Qualify built-in Postgres function names with pg_catalog if not schema-qualified
        def qualify_builtin(name):
            if name and "." not in name and name.lower() in self.PG_CATALOG_FUNCTIONS:
                return f"pg_catalog.{name}"
            return name

        _sfunc = qualify_builtin(_sfunc)
        _finalfunc = qualify_builtin(_finalfunc)

        # 3. Also rewrite function names inside the definition string
        definition = re.sub(
            r"SFUNC\s*=\s*(\w+)",
            lambda m: f"SFUNC = {qualify_builtin(m.group(1))}",
            definition,
            flags=re.IGNORECASE,
        )
        definition = re.sub(
            r"FINALFUNC\s*=\s*(\w+)",
            lambda m: f"FINALFUNC = {qualify_builtin(m.group(1))}",
            definition,
            flags=re.IGNORECASE,
        )

        # 4. Quote INITCOND as string for known integer types
        def quote_initcond_if_needed(def_str, stype):
            def replacer(match):
                value = match.group(1)
                if (
                    stype
                    and stype.lower() in self.KNOWN_INT_TYPES
                    and not (value.startswith("'") or value.startswith('"'))
                ):
                    return f"INITCOND = '{value}'"
                return f"INITCOND = {value}"

            return re.sub(r"INITCOND\s*=\s*([^\s,]+)", replacer, def_str, flags=re.IGNORECASE)

        definition = quote_initcond_if_needed(definition, _stype)

        # 5. Add any extra kwargs as aggregate options (e.g. PARALLEL)
        if kwargs:
            for key, value in kwargs.items():
                param = key.upper()
                if value is not None and f"{param} =" not in definition.upper():
                    definition += f",\n    {param} = {value}"

        # Clean up comments, double spaces, etc.
        cleaned = self.clean_aggregate_definition(definition)
        super().__init__(schema, signature, cleaned)

        self._sfunc = _sfunc
        self._stype = _stype
        self._finalfunc = _finalfunc
        self._initcond = _initcond
        self._stored_dependencies = _stored_dependencies or []

        # 6. Auto-fill dependencies if all_entities provided
        if all_entities is not None:
            self.autofill_dependencies(all_entities)

    @staticmethod
    def clean_aggregate_definition(definition: str) -> str:
        """
        Clean up aggregate definition string:
        - Remove comments (-- ...)
        - Replace multiple spaces/tabs with single space
        - Remove linebreaks
        - Beautify commas
        """
        definition = re.sub(r"--.*", "", definition)
        definition = re.sub(r"[ \t]+", " ", definition)
        definition = re.sub(r"\s*\n\s*", " ", definition)
        definition = re.sub(r"\s*,\s*", ", ", definition)
        return definition.strip()

    def autofill_initcond_for_type(self, stype):
        t = (stype or "").lower()
        defaults = {"text": "''"}

        return defaults.get(t) or ("0" if t in self.KNOWN_INT_TYPES else None)

    def autofill_dependencies(self, all_entities):
        """
        Tries to auto-detect dependencies (functions referenced by this aggregate)
        """
        dependency_names = [self._sfunc, self._finalfunc]
        self._stored_dependencies = [
            e
            for name in dependency_names
            if name
            for e in all_entities
            if getattr(e, "signature", "").startswith(name)
        ]

    @staticmethod
    def quote_ident(name):
        """Properly quote SQL identifiers if needed"""
        if not name.isidentifier() or name.lower() != name:
            return f'"{name}"'
        return name

    @classmethod
    def from_sql(cls, sql: str) -> "PGAggregate":
        """
        Create a PGAggregate instance from a SQL CREATE AGGREGATE statement.
        """
        normalized_sql = normalize_whitespace(sql.strip())
        pattern = r"create\s+aggregate\s+(?:(\w+)\.)?(\w+)\s*\(\s*([^)]*)\s*\)\s*\(\s*(.*?)\s*\)"
        match = re.search(pattern, normalized_sql, re.IGNORECASE | re.DOTALL)

        if match:
            schema_part = match.group(1) or "public"
            aggregate_name = match.group(2)
            parameters = match.group(3).strip()
            definition_body = match.group(4)
            signature = f"{aggregate_name}({parameters})" if parameters else f"{aggregate_name}()"
            return cls(
                schema=schema_part,
                signature=signature,
                definition=definition_body,
            )

        raise SQLParseFailure(f'Failed to parse SQL into PGAggregate """{sql}"""')

    @property
    def literal_signature(self) -> str:
        """
        Returns the aggregate signature with quoted name for SQL output.
        """
        name, remainder = self.signature.split("(", 1)
        return '"' + name.strip() + '"(' + remainder

    def _parse_aggregate_components(self) -> dict[str, str]:
        """
        Parses the aggregate definition string into a dict of components, e.g.
        SFUNC, STYPE, FINALFUNC, INITCOND, etc.
        """
        components = {}
        definition = self.definition.upper()
        patterns = {
            "SFUNC": r"SFUNC\s*=\s*([^,\s]+)",
            "STYPE": r"STYPE\s*=\s*([^,\s]+)",
            "FINALFUNC": r"FINALFUNC\s*=\s*([^,\s]+)",
            "INITCOND": r'INITCOND\s*=\s*[\'"]?([^,\'"]+)[\'"]?',
            "COMBINEFUNC": r"COMBINEFUNC\s*=\s*([^,\s]+)",
            "SERIALFUNC": r"SERIALFUNC\s*=\s*([^,\s]+)",
            "DESERIALFUNC": r"DESERIALFUNC\s*=\s*([^,\s]+)",
            "MSFUNC": r"MSFUNC\s*=\s*([^,\s]+)",
            "MINVFUNC": r"MINVFUNC\s*=\s*([^,\s]+)",
            "MSTYPE": r"MSTYPE\s*=\s*([^,\s]+)",
            "MSPACE": r"MSPACE\s*=\s*([^,\s]+)",
            "SORTOP": r"SORTOP\s*=\s*([^,\s]+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, definition)
            if match:
                components[key.lower()] = match.group(1).strip()
        return components

    def get_dependencies(self) -> List[str]:
        """
        Returns a list of referenced function signatures (as strings) for this aggregate.
        """

        dependencies = []
        components = self._parse_aggregate_components()
        function_components = [
            "sfunc",
            "finalfunc",
            "combinefunc",
            "serialfunc",
            "deserialfunc",
            "msfunc",
            "minvfunc",
        ]
        for comp in function_components:
            if comp in components:
                func_name = components[comp]
                # Add schema prefix if not qualified
                if "." not in func_name:
                    func_name = f"{self.schema}.{func_name}"
                dependencies.append(func_name)
        return dependencies

    def to_sql_statement_create(self):
        """
        Generate the CREATE AGGREGATE SQL statement for this object.
        """
        return sql_text(
            f"CREATE AGGREGATE {self.literal_schema}.{self.literal_signature} ({self.definition})"
        )

    def to_sql_statement_drop(self, cascade=False):
        """
        Generate the DROP AGGREGATE SQL statement for this object.
        """
        cascade_clause = " CASCADE" if cascade else ""
        template = "{aggregate_name}({parameters})"
        result = parse(template, self.signature, case_sensitive=False)
        try:
            aggregate_name = result["aggregate_name"].strip()
            parameters_str = result["parameters"].strip()
        except (TypeError, KeyError):
            result = parse("{aggregate_name}()", self.signature, case_sensitive=False)
            if result:
                aggregate_name = result["aggregate_name"].strip()
                parameters_str = ""
            else:
                aggregate_name = self.signature.split("(")[0].strip()
                parameters_str = ""
        # For DROP, only types are needed
        if parameters_str:
            param_types = []
            for param in parameters_str.split(","):
                param = param.strip()
                parts = param.split()
                param_types.append(parts[-1])
            drop_params = ", ".join(param_types)
        else:
            drop_params = ""
        return sql_text(
            f'DROP AGGREGATE {self.literal_schema}."{aggregate_name}"({drop_params}){cascade_clause}'
        )

    def to_sql_statement_create_or_replace(self) -> Generator[TextClause, Any, None]:
        """
        PostgreSQL does NOT support CREATE OR REPLACE AGGREGATE, so emit drop+create.
        """
        yield self.to_sql_statement_drop()
        yield self.to_sql_statement_create()

    @classmethod
    def from_database(cls, sess: Session, schema: str = "%") -> list["PGAggregate"]:  # type: ignore[override]
        """
        Returns all aggregates from the database as PGAggregate objects.
        """
        sql = sql_text(
            """SELECT
            n.nspname as schema,
            p.proname as agg_name,
            pg_catalog.pg_get_function_identity_arguments(p.oid) as signature_args,
            a.agginitval,
            SFUNC_NS.nspname || '.' || SFUNC.proname as sfunc,
            STYPE.typname as stype,
            FINALFUNC_NS.nspname || '.' || FINALFUNC.proname as finalfunc
        FROM pg_catalog.pg_aggregate a
        JOIN pg_catalog.pg_proc p ON p.oid = a.aggfnoid
        JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
        LEFT JOIN pg_catalog.pg_proc SFUNC ON SFUNC.oid = a.aggtransfn
        LEFT JOIN pg_catalog.pg_namespace SFUNC_NS ON SFUNC_NS.oid = SFUNC.pronamespace
        LEFT JOIN pg_catalog.pg_type STYPE ON STYPE.oid = a.aggtranstype
        LEFT JOIN pg_catalog.pg_proc FINALFUNC ON FINALFUNC.oid = a.aggfinalfn
        LEFT JOIN pg_catalog.pg_namespace FINALFUNC_NS ON FINALFUNC_NS.oid = FINALFUNC.pronamespace
        WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
        AND (n.nspname = :schema OR :schema = '%')
        ORDER BY n.nspname, p.proname
        """
        )
        rows = sess.execute(sql, {"schema": schema}).fetchall()
        db_aggregates = []
        for row in rows:
            try:
                if (
                    row[3]
                    and isinstance(row[3], str)
                    and row[3].strip().lower().startswith("create aggregate")
                ):
                    agg = cls.from_sql(row[3])
                else:
                    schema_name = row[0]
                    agg_name = row[1]
                    agg_args = row[2] or ""
                    initcond = row[3]
                    sfunc = row[4]
                    stype = row[5]
                    finalfunc = row[6]
                    parts = []
                    if sfunc:
                        parts.append(f"SFUNC = {sfunc}")
                    if stype:
                        parts.append(f"STYPE = {stype}")
                    if initcond is not None:
                        parts.append(f"INITCOND = '{initcond}'")
                    if finalfunc:
                        parts.append(f"FINALFUNC = {finalfunc}")
                    definition_body = ",\n".join(parts)
                    signature = f"{agg_name}({agg_args})"
                    agg = cls(
                        schema=schema_name,
                        signature=signature,
                        definition=definition_body,
                    )
                db_aggregates.append(agg)
            except Exception as e:
                try:
                    row_info = f"{row[0]}.{row[1]}"
                except Exception:
                    row_info = repr(row)
                print(f"Warning: Could not parse aggregate {row_info}: {e}")
                continue
        return db_aggregates

    @classmethod
    def render_import_statement(cls) -> str:
        """Render a string that is valid python code to import current class"""
        module_path = cls.__module__
        class_name = cls.__name__
        return f"from {module_path} import {class_name}"


# Legacy/compatibility helper for instantiation via old-style signature
def pg_aggregate_from_definition(
    schema: str, signature: str, definition: str, **kwargs
) -> PGAggregate:
    """Helper function to create PGAggregate with additional parameters"""
    return PGAggregate(schema=schema, signature=signature, definition=definition, **kwargs)
