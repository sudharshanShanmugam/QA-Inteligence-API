"""
Schema parser – converts DB schemas and API contracts into graph-ready entity specs.
Called during ingestion to augment entity extraction with structured schema data.
"""

from typing import Any, Dict, List


class SchemaParser:
    def parse_db_schema(self, schema_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse a DB schema dict into graph entities.
        Returns list of entity dicts ready for relationship_builder.
        """
        entities = []
        database = schema_data.get("database", "db")

        for table in schema_data.get("tables", []):
            tbl_name = table.get("name", "")
            tbl_id = f"tbl_{tbl_name.lower()}"

            cols = table.get("columns", [])
            pk_cols = [c["name"] for c in cols if c.get("primary_key")]
            fk_cols = [(c["name"], c["foreign_key"]) for c in cols if c.get("foreign_key")]
            nullable_cols = [c["name"] for c in cols if not c.get("not_null") and not c.get("primary_key")]

            entities.append({
                "id": tbl_id,
                "label": "DBTable",
                "name": tbl_name,
                "properties": {
                    "database": database,
                    "column_count": len(cols),
                    "primary_keys": pk_cols,
                    "foreign_keys": [fk[1] for fk in fk_cols],
                    "nullable_columns": nullable_cols[:10],
                    "description": table.get("description", ""),
                }
            })

        return entities

    def parse_api_contract(self, contract_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse an API contract dict into graph entities.
        """
        entities = []
        module_name = contract_data.get("module", "Unknown")
        module_id = f"mod_{module_name.lower().replace(' ', '_')}"

        # Module entity
        entities.append({
            "id": module_id,
            "label": "Module",
            "name": module_name,
            "properties": {"criticality": 4}
        })

        for ep in contract_data.get("endpoints", []):
            path = ep.get("path", "")
            method = ep.get("method", "GET")
            api_id = f"api_{method.lower()}_{path.lower().replace('/', '_').replace('{', '').replace('}', '').strip('_')}"

            entities.append({
                "id": api_id,
                "label": "API",
                "name": f"{method} {path}",
                "properties": {
                    "endpoint": path,
                    "method": method,
                    "module_id": module_id,
                    "description": ep.get("description", ""),
                    "auth_required": ep.get("auth_required", True),
                    "idempotency": ep.get("idempotency", False),
                    "events_published": ep.get("events_published", []),
                    "rate_limit": ep.get("rate_limit", ""),
                }
            })

            # Event entities for published events
            for ev_name in ep.get("events_published", []):
                ev_id = f"event_{ev_name.lower()}"
                entities.append({
                    "id": ev_id,
                    "label": "Event",
                    "name": ev_name,
                    "properties": {"producer": module_name}
                })

        return entities

    def extract_field_constraints(self, contract_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract field-level constraints from API request bodies for BVA/EP engine.
        Returns a list of field specs.
        """
        fields = []
        for ep in contract_data.get("endpoints", []):
            req_body = ep.get("request_body", {})
            for field_name, field_desc in req_body.items():
                field_spec = self._infer_field_spec(field_name, str(field_desc))
                if field_spec:
                    fields.append(field_spec)
        return fields

    @staticmethod
    def _infer_field_spec(name: str, description: str) -> Dict[str, Any]:
        """Infer field type and constraints from name + description string."""
        name_lower = name.lower()
        desc_lower = description.lower()

        spec: Dict[str, Any] = {"name": name, "required": "optional" not in desc_lower}

        if "uuid" in desc_lower:
            spec["type"] = "string"
            spec["min"] = 36
            spec["max"] = 36
        elif "email" in name_lower:
            spec["type"] = "email"
        elif "phone" in name_lower or "mobile" in name_lower:
            spec["type"] = "phone"
        elif any(w in name_lower for w in ["amount", "price", "total", "cost"]):
            spec["type"] = "float"
            spec["min"] = 0.01
            spec["max"] = 999999.99
        elif any(w in name_lower for w in ["count", "qty", "quantity", "number"]):
            spec["type"] = "integer"
            spec["min"] = 0
            spec["max"] = 10000
        elif "date" in name_lower or "timestamp" in name_lower:
            spec["type"] = "date"
        elif "boolean" in desc_lower or name_lower.startswith("is_") or name_lower.startswith("has_"):
            spec["type"] = "boolean"
        elif "max" in desc_lower:
            import re
            m = re.search(r'max[:\s]+(\d+)', desc_lower)
            if m:
                spec["type"] = "string"
                spec["max"] = int(m.group(1))
            else:
                spec["type"] = "string"
        else:
            spec["type"] = "string"

        return spec


schema_parser = SchemaParser()
