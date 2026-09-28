"""Banking tools (engine Tool contract: invoke mutates data / returns str).

Three request types share ONE decision protocol: decide(case_id, ...) writes
the outcome onto the case record, so the DB hash grades decision AND reason.
Decisions are ONE-SHOT: a case that has left `pending` can never be
re-decided, so answers cannot be found by enumeration.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from ...engine.tool import Tool
from .l1.rules import REASONS

CASE_TABLES = ("transactions", "limit_requests", "transfers")


def _find_case(data: Dict[str, Any], case_id: str):
    for tbl in CASE_TABLES:
        if case_id in data.get(tbl, {}):
            return data[tbl][case_id]
    return None


class GetCaseDetails(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], case_id: str) -> str:
        c = _find_case(data, case_id)
        if c is None:
            return "Error: case not found"
        return json.dumps(c)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {"type": "function", "function": {
            "name": "get_case_details",
            "description": "Look up the case under review — a pending card "
                           "transaction (TXN-...), a limit-increase request "
                           "(LIM-...) or an outbound transfer (TRF-...).",
            "parameters": {"type": "object", "properties": {
                "case_id": {"type": "string"}}, "required": ["case_id"]}}}


class GetAccountDetails(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], account_id: str) -> str:
        acc = data["accounts"].get(account_id)
        if acc is None:
            return "Error: account not found"
        return json.dumps(acc)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {"type": "function", "function": {
            "name": "get_account_details",
            "description": "Look up an account: holder, home state, verification "
                           "level, standing, tenure, current credit limit.",
            "parameters": {"type": "object", "properties": {
                "account_id": {"type": "string"}},
                "required": ["account_id"]}}}


class GetMerchantDetails(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], merchant_id: str) -> str:
        m = data["merchants"].get(merchant_id)
        if m is None:
            return "Error: merchant not found"
        return json.dumps(m)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {"type": "function", "function": {
            "name": "get_merchant_details",
            "description": "Look up a merchant: name, category, region.",
            "parameters": {"type": "object", "properties": {
                "merchant_id": {"type": "string"}},
                "required": ["merchant_id"]}}}


class GetPayeeDetails(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], payee_id: str) -> str:
        p = data["payees"].get(payee_id)
        if p is None:
            return "Error: payee not found"
        return json.dumps(p)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {"type": "function", "function": {
            "name": "get_payee_details",
            "description": "Look up a transfer payee: name, type, "
                           "jurisdiction.",
            "parameters": {"type": "object", "properties": {
                "payee_id": {"type": "string"}}, "required": ["payee_id"]}}}


def _decide(data, case_id, status, reason):
    c = _find_case(data, case_id)
    if c is None:
        return "Error: case not found"
    if c["status"] != "pending":
        return f"Error: case already {c['status']} — decisions are final"
    c["status"] = status
    c["reason"] = reason
    return f"Case {case_id} {status}" + (f" ({reason})" if reason else "")


class ApproveCase(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], case_id: str) -> str:
        return _decide(data, case_id, "approved", None)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {"type": "function", "function": {
            "name": "approve_case",
            "description": "Approve the pending case. Final.",
            "parameters": {"type": "object", "properties": {
                "case_id": {"type": "string"}}, "required": ["case_id"]}}}


class DenyCase(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], case_id: str, reason: str) -> str:
        if reason not in REASONS:
            return "Error: invalid reason code"
        return _decide(data, case_id, "denied", reason)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {"type": "function", "function": {
            "name": "deny_case",
            "description": "Deny the pending case, citing a reason code. Final.",
            "parameters": {"type": "object", "properties": {
                "case_id": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["case_id", "reason"]}}}


class EscalateCase(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], case_id: str, reason: str) -> str:
        if reason not in REASONS:
            return "Error: invalid reason code"
        return _decide(data, case_id, "escalated", reason)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {"type": "function", "function": {
            "name": "escalate_case",
            "description": "Send the pending case to manual review, citing a "
                           "reason code. Final.",
            "parameters": {"type": "object", "properties": {
                "case_id": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["case_id", "reason"]}}}
