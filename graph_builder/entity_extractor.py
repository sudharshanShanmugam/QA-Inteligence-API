"""
Entity extraction from raw documents — rule-based, instant, no LLM.

Extracts: Module, Feature, API, DBTable, Event, TestCase, Bug,
          BusinessRule, State — with their properties and relationships.
"""

import re
from typing import Any, Dict, List
import structlog

log = structlog.get_logger()


def _slug(text: str) -> str:
    return re.sub(r'\W+', '_', text.lower().strip()).strip('_')[:60]


class EntityExtractor:

    def extract(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities and relationships from a document — rule-based, instant."""
        doc_type = document.get("doc_type", "")

        handlers = {
            "bug_report":        self._extract_bug_report,
            "api_contract":      self._extract_api_contract,
            "db_schema":         self._extract_db_schema,
            "user_story":        self._extract_user_story,
            "brd":               self._extract_brd_srs,
            "srs":               self._extract_brd_srs,
            "test_case":         self._extract_test_case,
            "event_definition":  self._extract_event,
            "business_rule":     self._extract_business_rule,
        }

        handler = handlers.get(doc_type, self._extract_generic)
        result = handler(document)

        result["entities"]      = [self._normalize(e) for e in result.get("entities", [])]
        result["relationships"] = [r for r in result.get("relationships", [])
                                   if r.get("from_id") and r.get("to_id")]

        log.info("entities_extracted",
                 source_id=document.get("source_id"),
                 doc_type=doc_type,
                 entities=len(result["entities"]),
                 relationships=len(result["relationships"]))
        return result

    # ── Bug report ────────────────────────────────────────────────────────────

    def _extract_bug_report(self, document: Dict[str, Any]) -> Dict[str, Any]:
        entities: List[Dict] = []
        relationships: List[Dict] = []
        content = document.get("content", "")
        raw = document.get("raw", {})
        source_id = document.get("source_id", "doc")

        # Handle JSON-structured bug reports
        if raw:
            return self._extract_bug_from_raw(raw, source_id)

        # Handle plain-text bug reports — parse line by line
        bugs = self._parse_text_bugs(content, source_id)
        for bug_entity, mod_entity, feat_entity, rels in bugs:
            entities.append(bug_entity)
            if mod_entity:
                entities.append(mod_entity)
            if feat_entity:
                entities.append(feat_entity)
            relationships.extend(rels)

        return {"entities": entities, "relationships": relationships}

    def _extract_bug_from_raw(self, raw: dict, source_id: str):
        entities, relationships = [], []
        bug_id = f"bug_{_slug(source_id)}"
        entities.append({
            "id": bug_id, "label": "Bug",
            "name": raw.get("title", source_id),
            "properties": {
                "severity":   raw.get("severity", "medium"),
                "status":     raw.get("status", "open"),
                "root_cause": raw.get("root_cause", ""),
                "description":raw.get("description", ""),
            }
        })
        if raw.get("module"):
            mid = f"mod_{_slug(raw['module'])}"
            entities.append({"id": mid, "label": "Module", "name": raw["module"],
                              "properties": {"criticality": 3}})
            relationships.append({"from_id": bug_id, "to_id": mid, "type": "FOUND_IN", "properties": {}})
        if raw.get("feature"):
            fid = f"feat_{_slug(raw['feature'])}"
            entities.append({"id": fid, "label": "Feature", "name": raw["feature"], "properties": {}})
            relationships.append({"from_id": bug_id, "to_id": fid, "type": "RELATED_TO", "properties": {}})
        return {"entities": entities, "relationships": relationships}

    def _parse_text_bugs(self, content: str, source_id: str):
        """Parse plain-text bug reports. Handles multi-bug files."""
        results = []
        # Split on BUG-NNN or numbered bug blocks
        blocks = re.split(r'(?=BUG[-_]\d+|DEFECT[-_]\d+|\n#{1,3}\s)', content, flags=re.IGNORECASE)
        if len(blocks) <= 1:
            blocks = [content]

        for i, block in enumerate(blocks):
            if not block.strip():
                continue
            bug_id = f"bug_{_slug(source_id)}_{i+1}"

            # Try to extract a bug ID from the block
            id_match = re.search(r'(BUG[-_]\d+|DEFECT[-_]\d+|ISSUE[-_]\d+)', block, re.IGNORECASE)
            if id_match:
                bug_id = f"bug_{_slug(id_match.group(1))}"

            # Title — first non-empty line or after the ID
            lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
            title = lines[0] if lines else source_id
            title = re.sub(r'^(BUG|DEFECT|ISSUE)[-_]\d+[:\s]*', '', title, flags=re.IGNORECASE).strip() or title

            severity = self._extract_field(block, r'severity[:\s]+(\w+)', "medium")
            status   = self._extract_field(block, r'status[:\s]+(\w[\w\s]*?)(?:\n|$)', "open")
            root_cause = self._extract_field(block, r'root\s*cause[:\s]+(.+?)(?:\n|$)', "")
            module   = self._extract_field(block, r'module[:\s]+(.+?)(?:\n|$)', "")
            feature  = self._extract_field(block, r'feature[:\s]+(.+?)(?:\n|$)', "")

            bug_entity = {
                "id": bug_id, "label": "Bug", "name": title,
                "properties": {"severity": severity.lower(), "status": status.lower(),
                               "root_cause": root_cause}
            }
            mod_entity, feat_entity, rels = None, None, []

            if module:
                mid = f"mod_{_slug(module)}"
                mod_entity = {"id": mid, "label": "Module", "name": module,
                              "properties": {"criticality": 3}}
                rels.append({"from_id": bug_id, "to_id": mid, "type": "FOUND_IN", "properties": {}})

            if feature:
                fid = f"feat_{_slug(feature)}"
                feat_entity = {"id": fid, "label": "Feature", "name": feature, "properties": {}}
                rels.append({"from_id": bug_id, "to_id": fid, "type": "RELATED_TO", "properties": {}})

            results.append((bug_entity, mod_entity, feat_entity, rels))

        return results

    # ── API contract ──────────────────────────────────────────────────────────

    def _extract_api_contract(self, document: Dict[str, Any]) -> Dict[str, Any]:
        entities, relationships = [], []
        raw = document.get("raw", {})
        content = document.get("content", "")
        source_id = document.get("source_id", "api")

        # Detect or create a module for this API document
        module_name = self._detect_module_from_text(content) or source_id.replace("_", " ").title()
        mod_id = f"mod_{_slug(module_name)}"
        entities.append({"id": mod_id, "label": "Module", "name": module_name,
                         "properties": {"criticality": 3}})

        if raw and raw.get("endpoints"):
            for ep in raw["endpoints"]:
                api_id = f"api_{_slug(ep.get('path', ''))}"
                entities.append({
                    "id": api_id, "label": "API",
                    "name": f"{ep.get('method','GET')} {ep.get('path','')}",
                    "properties": {"endpoint": ep.get("path",""), "method": ep.get("method","GET"),
                                   "description": ep.get("description","")}
                })
                relationships.append({"from_id": api_id, "to_id": mod_id,
                                      "type": "BELONGS_TO", "properties": {}})
        else:
            # Parse plain-text — look for HTTP method + path patterns
            for match in re.finditer(r'\b(GET|POST|PUT|PATCH|DELETE)\s+(/[\w/{}._-]+)',
                                     content, re.IGNORECASE):
                method, path = match.group(1).upper(), match.group(2)
                api_id = f"api_{_slug(path)}"
                if not any(e["id"] == api_id for e in entities):
                    entities.append({
                        "id": api_id, "label": "API",
                        "name": f"{method} {path}",
                        "properties": {"endpoint": path, "method": method}
                    })
                    relationships.append({"from_id": api_id, "to_id": mod_id,
                                          "type": "BELONGS_TO", "properties": {}})

        return {"entities": entities, "relationships": relationships}

    # ── DB schema ─────────────────────────────────────────────────────────────

    def _extract_db_schema(self, document: Dict[str, Any]) -> Dict[str, Any]:
        entities, relationships = [], []
        raw = document.get("raw", {})
        content = document.get("content", "")

        if raw and raw.get("tables"):
            for table in raw["tables"]:
                tbl_id = f"tbl_{_slug(table.get('name',''))}"
                entities.append({
                    "id": tbl_id, "label": "DBTable", "name": table.get("name",""),
                    "properties": {"columns": len(table.get("columns", []))}
                })
        else:
            # Detect CREATE TABLE statements or table mentions
            for match in re.finditer(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"]?(\w+)[`"]?', content, re.IGNORECASE):
                tbl_id = f"tbl_{_slug(match.group(1))}"
                if not any(e["id"] == tbl_id for e in entities):
                    entities.append({"id": tbl_id, "label": "DBTable", "name": match.group(1),
                                     "properties": {}})

        return {"entities": entities, "relationships": relationships}

    # ── User story ────────────────────────────────────────────────────────────

    def _extract_user_story(self, document: Dict[str, Any]) -> Dict[str, Any]:
        entities, relationships = [], []
        content = document.get("content", "")
        source_id = document.get("source_id", "story")
        raw = document.get("raw", {})

        # Feature name — from "I want to ..." or title
        feat_name = ""
        if raw.get("goal"):
            feat_name = raw["goal"]
        else:
            m = re.search(r'i want (?:to )?(.+?)(?:so that|,|\.|$)', content.lower())
            feat_name = m.group(1).strip()[:80] if m else source_id

        feat_id = f"feat_{_slug(feat_name)}"
        entities.append({
            "id": feat_id, "label": "Feature", "name": feat_name,
            "properties": {"priority": "medium", "source": source_id}
        })

        # Module — from "as a ... in <module>" or keyword detection
        module = self._detect_module_from_text(content)
        if module:
            mid = f"mod_{_slug(module)}"
            entities.append({"id": mid, "label": "Module", "name": module,
                              "properties": {"criticality": 3}})
            relationships.append({"from_id": feat_id, "to_id": mid, "type": "BELONGS_TO", "properties": {}})

        # Business rules from acceptance criteria
        ac_block = re.search(r'acceptance criteria[:\s]*(.*?)(?:\n\n|test data|$)', content,
                             re.IGNORECASE | re.DOTALL)
        if ac_block:
            for i, line in enumerate(ac_block.group(1).strip().split('\n'), 1):
                line = re.sub(r'^[-*\d.AC:]+\s*', '', line).strip()
                if len(line) > 10:
                    br_id = f"br_{_slug(source_id)}_{i}"
                    entities.append({"id": br_id, "label": "BusinessRule", "name": line[:120],
                                     "properties": {"source": source_id}})
                    relationships.append({"from_id": br_id, "to_id": feat_id,
                                          "type": "GOVERNED_BY", "properties": {}})

        # States mentioned — link each to the Feature
        state_words = re.findall(
            r'\b([A-Z][A-Z_]{3,})\b|'
            r'\b(pending|active|failed|cancelled|completed|processing|confirmed|approved|rejected)\b',
            content)
        seen_states = set()
        for groups in state_words:
            state = (groups[0] or groups[1]).upper()
            if state not in seen_states and len(state) > 3:
                seen_states.add(state)
                sid = f"state_{_slug(state)}"
                entities.append({"id": sid, "label": "State", "name": state, "properties": {}})
                relationships.append({"from_id": feat_id, "to_id": sid,
                                      "type": "TRANSITIONS_TO", "properties": {}})

        return {"entities": entities, "relationships": relationships}

    # ── BRD / SRS ─────────────────────────────────────────────────────────────

    def _extract_brd_srs(self, document: Dict[str, Any]) -> Dict[str, Any]:
        entities, relationships = [], []
        content = document.get("content", "")
        source_id = document.get("source_id", "doc")

        last_module_id = None
        last_feature_id = None
        seen_ids: set = set()

        # Single sequential pass — each line is processed in document order
        # so BR-001 always links to the feature above it, not the last feature in the file
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # ── Header line? ─────────────────────────────────────────────────
            h = re.match(r'^(#{1,4}|[0-9]+\.[0-9]*\.?)\s+(.+)$', line)
            if h:
                marker, header = h.group(1).strip(), h.group(2).strip()
                if len(header) < 3 or len(header) > 100:
                    continue
                is_top = re.match(r'^#$', marker) or re.match(r'^[0-9]+\.$', marker)
                if is_top:
                    mid = f"mod_{_slug(header)}"
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        entities.append({"id": mid, "label": "Module", "name": header,
                                         "properties": {"criticality": 3, "source": source_id}})
                    last_module_id = mid
                    last_feature_id = None
                else:
                    fid = f"feat_{_slug(header)}"
                    if fid not in seen_ids:
                        seen_ids.add(fid)
                        entities.append({"id": fid, "label": "Feature", "name": header,
                                         "properties": {"priority": "medium"}})
                        if last_module_id:
                            relationships.append({"from_id": fid, "to_id": last_module_id,
                                                  "type": "BELONGS_TO", "properties": {}})
                    last_feature_id = fid
                continue

            # ── Business rule line? ───────────────────────────────────────────
            br = re.match(r'(BR[-_]\w+|Rule\s+\d+)[:\s]+(.+)', line, re.IGNORECASE)
            if br:
                br_id = f"br_{_slug(br.group(1))}"
                if br_id not in seen_ids:
                    seen_ids.add(br_id)
                    entities.append({"id": br_id, "label": "BusinessRule",
                                     "name": br.group(2).strip()[:120],
                                     "properties": {"rule_id": br.group(1)}})
                    anchor_id = last_feature_id or last_module_id
                    if anchor_id:
                        relationships.append({"from_id": br_id, "to_id": anchor_id,
                                              "type": "GOVERNED_BY", "properties": {}})
                continue

            # ── API endpoint line? ────────────────────────────────────────────
            api = re.search(r'\b(GET|POST|PUT|PATCH|DELETE)\s+(/[\w/{}._-]+)', line, re.IGNORECASE)
            if api:
                api_id = f"api_{_slug(api.group(2))}"
                if api_id not in seen_ids:
                    seen_ids.add(api_id)
                    entities.append({"id": api_id, "label": "API",
                                     "name": f"{api.group(1).upper()} {api.group(2)}",
                                     "properties": {"endpoint": api.group(2),
                                                    "method": api.group(1).upper()}})
                    anchor_id = last_feature_id or last_module_id
                    if anchor_id:
                        relationships.append({"from_id": api_id, "to_id": anchor_id,
                                              "type": "BELONGS_TO", "properties": {}})

        return {"entities": entities, "relationships": relationships}

    # ── Test case ─────────────────────────────────────────────────────────────

    def _extract_test_case(self, document: Dict[str, Any]) -> Dict[str, Any]:
        entities, relationships = [], []
        content = document.get("content", "")
        source_id = document.get("source_id", "tc")

        # Match TC-NNN or Test Case N patterns
        for match in re.finditer(r'(TC[-_]\d+|Test\s+Case\s+\d+)[:\s]+(.+?)(?:\n|$)', content, re.IGNORECASE):
            tc_id = f"tc_{_slug(match.group(1))}"
            entities.append({
                "id": tc_id, "label": "TestCase",
                "name": match.group(2).strip()[:120],
                "properties": {"test_id": match.group(1), "source": source_id}
            })

        return {"entities": entities, "relationships": relationships}

    # ── Event definition ──────────────────────────────────────────────────────

    def _extract_event(self, document: Dict[str, Any]) -> Dict[str, Any]:
        entities, relationships = [], []
        raw = document.get("raw", {})
        content = document.get("content", "")

        if raw and raw.get("name"):
            ev_id = f"evt_{_slug(raw['name'])}"
            entities.append({
                "id": ev_id, "label": "Event", "name": raw["name"],
                "properties": {"topic": raw.get("topic", ""), "type": raw.get("type", "")}
            })
        else:
            for match in re.finditer(r'(?:event|topic|message)[:\s]+["\']?([\w._-]+)["\']?', content, re.IGNORECASE):
                ev_id = f"evt_{_slug(match.group(1))}"
                if not any(e["id"] == ev_id for e in entities):
                    entities.append({"id": ev_id, "label": "Event", "name": match.group(1),
                                     "properties": {}})

        return {"entities": entities, "relationships": relationships}

    # ── Business rule ─────────────────────────────────────────────────────────

    def _extract_business_rule(self, document: Dict[str, Any]) -> Dict[str, Any]:
        entities, relationships = [], []
        content = document.get("content", "")

        for i, match in enumerate(re.finditer(r'(?:BR[-_]\w+|Rule\s+\d+)[:\s]+(.+?)(?:\n|$)', content, re.IGNORECASE), 1):
            br_id = f"br_{i:03d}"
            entities.append({"id": br_id, "label": "BusinessRule",
                             "name": match.group(1).strip()[:120], "properties": {}})

        return {"entities": entities, "relationships": relationships}

    # ── Generic fallback ──────────────────────────────────────────────────────

    def _extract_generic(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Generic extraction for any doc type — pulls modules and features from headers."""
        entities, relationships = [], []
        content = document.get("content", "")

        module = self._detect_module_from_text(content)
        if module:
            mid = f"mod_{_slug(module)}"
            entities.append({"id": mid, "label": "Module", "name": module,
                              "properties": {"criticality": 3}})

        for match in re.finditer(r'^#{1,3}\s+(.+)$', content, re.MULTILINE):
            header = match.group(1).strip()
            if 3 < len(header) < 80:
                fid = f"feat_{_slug(header)}"
                if not any(e["id"] == fid for e in entities):
                    entities.append({"id": fid, "label": "Feature", "name": header,
                                     "properties": {}})

        return {"entities": entities, "relationships": relationships}

    # ── Shared helpers ────────────────────────────────────────────────────────

    _MODULE_KEYWORDS = {
        "payment": "Payment", "checkout": "Checkout", "order": "Orders",
        "cart": "Cart", "auth": "Authentication", "login": "Authentication",
        "user": "User Management", "profile": "User Management",
        "notification": "Notifications", "email": "Notifications",
        "report": "Reporting", "dashboard": "Dashboard", "search": "Search",
        "product": "Product Catalogue", "inventory": "Inventory",
        "shipping": "Shipping", "upload": "Document Management",
        "document": "Document Management", "enroll": "Enrollment",
        "course": "Learning", "certificate": "Certification",
    }

    def _detect_module_from_text(self, text: str) -> str:
        text_lower = text.lower()
        for kw, module in self._MODULE_KEYWORDS.items():
            if kw in text_lower:
                return module
        return ""

    @staticmethod
    def _extract_field(text: str, pattern: str, default: str) -> str:
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else default

    @staticmethod
    def _normalize(entity: Dict[str, Any]) -> Dict[str, Any]:
        if not entity.get("id"):
            entity["id"] = _slug(entity.get("name", "unknown"))
        if not entity.get("name"):
            entity["name"] = entity["id"]
        entity.setdefault("properties", {})
        return entity


entity_extractor = EntityExtractor()
