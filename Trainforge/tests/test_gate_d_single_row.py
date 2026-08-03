"""Hermetic adversarials for the reviewed one-row Gate-D authority."""
from __future__ import annotations
import copy, json
import threading
from pathlib import Path
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from lib.ontology.misconception_id import canonical_mc_id
from Trainforge.scripts.archive.gate_d_single_row import (
    CAPABILITY_SCHEMA, CONTROL_EVIDENCE_SCHEMA, GATE_A_TRUST_SCHEMA,
    PINNED_AUTHORITY_SHA256, PINNED_CONTRACT_SHA256,
    PINNED_GO_CANONICAL_SHA256, PINNED_RELEASE_ROOT_SHA256, PINNED_ROW,
    PINNED_TUPLE_SHA256, SIGNED_WRAPPER_SCHEMA, GateDCallController,
    SecureOutputTree, _sha, _stable, authorize_single_row, row_identity,
    _offline_pair_validator, PAIR_AUDIT_FIELDS_SCHEMA_ID,
    PREFERENCE_PAIR_SCHEMA_ID,
    authorize_functional_single_row, collect_functional_preflight,
    collect_functional_postflight, functional_reasoning_bytes,
    write_unconsumed, verify_gate_d_precommit, verify_full_gate_d_transaction,
)

H = "a" * 64
GO = Path("plans/release-evidence/training-synthesis-release-v1.2.3/"
          "04-independent-go/gate-d-go.json").resolve()
MANIFEST="655b0dfc2e396c09ca1ae5bcc7a078a854ee9e2064c97ab4d2210bf7c0b00936"
ELIG="95fd0ba42871968fb344961669299fc70643b94bcff2b6b3922c5117dddc018a"
ORDER="f0be5cf9600418e60e75b0a566234f51118f64b94ca421686322e203d456315e"
RUN="gate-d-v1.2.3-dpo-00276-seed0-001"
OUT=Path("plans/release-evidence/training-synthesis-release-v1.2.3/"
 "05-one-dpo-canary/gate-d-v1.2.3-dpo-00276-seed0-001").resolve()
FUNCTIONAL_ROW_ID="openstax_ea2e_scan_eval_chunk_00183"
FUNCTIONAL_ROW_SHA="44082f134734ef019e01f0bc51cfa887725fde67089e1146cd00f29b5b4452c0"


def _local_pair_schema_fixture(tmp_path):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    source = Path("schemas/knowledge")
    for name in ("preference_pair.schema.json", "pair_audit_fields.schema.json"):
        (knowledge / name).write_bytes((source / name).read_bytes())
    return knowledge


def test_gate_d_pair_registry_resolves_nested_per_claim_support_offline(
    tmp_path, monkeypatch,
):
    knowledge = _local_pair_schema_fixture(tmp_path)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("network retrieval is forbidden"),
    )
    validator, evidence = _offline_pair_validator(
        knowledge / "preference_pair.schema.json",
        knowledge_dir=knowledge,
    )
    pair = {
        "prompt": "Analyze the supplied evidence without unsupported claims.",
        "chosen": "The supported conclusion follows from the supplied evidence and its stated relationship.",
        "rejected": "Every possible conclusion follows, even when the supplied evidence does not support it.",
        "chunk_id": "generic-chunk",
        "lo_refs": ["co-generic"],
        "seed": 0,
        "decision_capture_id": "EVT_generic",
        "per_claim_support": [{
            "sentence": "The supported conclusion follows.",
            "entailment": 0.9,
            "contradiction": 0.01,
            "outcome": "entailed",
            "source_chunk_ids": None,
        }],
    }
    validator.validate(pair)
    assert evidence["root_id"] == PREFERENCE_PAIR_SCHEMA_ID
    assert PAIR_AUDIT_FIELDS_SCHEMA_ID in evidence["resources"]
    assert len(evidence["sha256"]) == 64


@pytest.mark.parametrize(
    "mutation",
    ("missing", "wrong_id", "duplicate_id", "malformed"),
)
def test_gate_d_pair_registry_fails_closed_on_local_authority_drift(
    tmp_path, mutation,
):
    knowledge = _local_pair_schema_fixture(tmp_path)
    audit = knowledge / "pair_audit_fields.schema.json"
    if mutation == "missing":
        audit.unlink()
    elif mutation == "wrong_id":
        payload = json.loads(audit.read_text())
        payload["$id"] = "https://ed4all.dev/ns/knowledge/v1/wrong.json"
        audit.write_text(json.dumps(payload))
    elif mutation == "duplicate_id":
        duplicate = json.loads(audit.read_text())
        (knowledge / "duplicate.schema.json").write_text(json.dumps(duplicate))
    else:
        audit.write_text("{")
    with pytest.raises(ValueError):
        _offline_pair_validator(
            knowledge / "preference_pair.schema.json",
            knowledge_dir=knowledge,
        )


def test_gate_d_pair_registry_unknown_ref_never_uses_network(
    tmp_path, monkeypatch,
):
    knowledge = _local_pair_schema_fixture(tmp_path)
    root = knowledge / "preference_pair.schema.json"
    schema = json.loads(root.read_text())
    schema["properties"]["unknown"] = {
        "$ref": "https://ed4all.dev/ns/knowledge/v1/unknown.schema.json"
    }
    root.write_text(json.dumps(schema))
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("network retrieval is forbidden"),
    )
    validator, _evidence = _offline_pair_validator(
        root, knowledge_dir=knowledge,
    )
    with pytest.raises(Exception, match="unknown.schema.json"):
        validator.validate({"unknown": {}})


def test_gate_d_pair_registry_existing_invalid_pair_still_fails(tmp_path):
    from jsonschema import ValidationError

    knowledge = _local_pair_schema_fixture(tmp_path)
    validator, _evidence = _offline_pair_validator(
        knowledge / "preference_pair.schema.json",
        knowledge_dir=knowledge,
    )
    with pytest.raises(ValidationError):
        validator.validate({
            "prompt": "too short",
            "chosen": "",
            "rejected": "",
        })

def rows():
    result=[{"chunk_id":f"c{i}","chunk_sha256":f"{i:064x}","kind":"instruction",
             "variant":"micro","repetition":0,
             "focus_objective":{"id":f"co-{i}"}} for i in range(8)]
    result[3].update({"chunk_id":PINNED_ROW["chunk_id"],
      "chunk_sha256":PINNED_ROW["chunk_sha256"],"kind":"preference",
      "variant":PINNED_ROW["variant"],"focus_objective":{"id":"co-155"}})
    return result

def functional_rows():
    result=rows()
    focus={"id":"co-1","statement":"Analyze how atomicity rolls back partial updates.",
      "bloom_level":"analyze"}
    misconception="atomic transactions preserve partial success"
    correction="atomicity rolls back partial updates"
    chunk={"id":FUNCTIONAL_ROW_ID,"text":(
      "A common misconception is that atomic transactions preserve partial "
      "success. The correction is that atomicity rolls back partial updates, "
      "so every operation succeeds together or every operation fails."),
      "learning_outcome_refs":["co-1"],"synthesis_focus_objective":focus,
      "html":(
        '<p data-cf-block-id="misconception-claim-1" '
        'data-cf-objective-id="co-1">Atomic transactions preserve partial '
        'success.</p><p data-cf-block-id="misconception-correction-1" '
        'data-cf-objective-id="co-1">Atomicity rolls back partial updates, '
        'so every operation succeeds together or every operation fails.</p>'
      ),
      "misconceptions":[{"id":canonical_mc_id(
          misconception, correction, "analyze"),
        "misconception":misconception,
        "correction":correction,
        "mechanism_evidence":"partial success conflicts with all-or-none"}]}
    result[3].update({"chunk_id":FUNCTIONAL_ROW_ID,
      "chunk_sha256":FUNCTIONAL_ROW_SHA,"kind":"preference",
      "variant":"D_production_contract","focus_objective":focus,"_chunk":chunk})
    return result

def signed(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    key=Ed25519PrivateKey.generate()
    raw=key.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)
    trust=tmp_path/"wp11.pub"; trust.write_bytes(raw); trust.chmod(0o600)
    authority=tmp_path/"gate-a.json"; authority.write_text(json.dumps({
      "schema":GATE_A_TRUST_SCHEMA,"tuple_sha256":PINNED_TUPLE_SHA256,
      "authority_sha256":PINNED_AUTHORITY_SHA256,
      "wp11_public_key_sha256":__import__("hashlib").sha256(raw).hexdigest(),
      "trusted_output_root":str(OUT.parent)}))
    authority.chmod(0o600)
    observed=__import__("time").monotonic_ns()
    evidence={"schema":CONTROL_EVIDENCE_SCHEMA,"endpoint":"http://control/v1",
      "served_model":"super","model_revision":"rev","backend":"trtllm",
      "thinking_enabled":False,"observed_monotonic_ns":observed}
    ep=tmp_path/"control-plane-evidence.json"; ep.write_text(json.dumps(evidence)); ep.chmod(0o600)
    proof={"served_model":"super","model_revision":"rev","backend":"trtllm",
      "backend_version":"1","dialect":"openai_json_schema_strict",
      "schema_sha256":H,"max_context_tokens":32768,"max_output_tokens":2048,
      "thinking_enabled":False,"endpoint":"http://control/v1",
      "control_evidence_sha256":__import__("hashlib").sha256(ep.read_bytes()).hexdigest(),
      "observed_monotonic_ns":observed,"fresh_for_ns":60_000_000_000}
    capability={"schema":CAPABILITY_SCHEMA,"proof":proof,
      "signature_hex":key.sign(_stable(proof).encode()).hex()}
    cap=tmp_path/"cap.json"; cap.write_text(json.dumps(capability)); cap.chmod(0o600)
    go_bytes=GO.read_bytes()
    go_copy=tmp_path/"gate-d-go.json"; go_copy.write_bytes(go_bytes); go_copy.chmod(0o600)
    payload={"go_canonical_sha256":PINNED_GO_CANONICAL_SHA256,
      "go_file_sha256":__import__("hashlib").sha256(go_bytes).hexdigest(),
      "release_root_sha256":PINNED_RELEASE_ROOT_SHA256,
      "authority_sha256":PINNED_AUTHORITY_SHA256,
      "tuple_sha256":PINNED_TUPLE_SHA256,"contract_sha256":PINNED_CONTRACT_SHA256,
      "capability_file_sha256":__import__("hashlib").sha256(cap.read_bytes()).hexdigest()}
    wrapper={"schema":SIGNED_WRAPPER_SCHEMA,"payload":payload,
      "signature_hex":key.sign(_stable(payload).encode()).hex()}
    wp=tmp_path/"wrapper.json"; wp.write_text(json.dumps(wrapper)); wp.chmod(0o600)
    return go_copy,wp,cap,trust,authority

def auth(tmp_path, rs=None, mutate=None):
    rs=rs or rows(); go,wp,cap,trust,authority=signed(tmp_path)
    values={"full_manifest_sha256":MANIFEST,"eligibility_sha256":ELIG,
      "ordered_identity_sha256":ORDER,"synthesis_seed":0,"run_id":RUN,
      "output_dir":OUT,"go_path":go,"wrapper_path":wp,"capability_path":cap,
      "trust_root_path":trust,"gate_a_authority_path":authority}
    if mutate: mutate(values)
    return authorize_single_row(rows=rs,**values)

def test_full8_validated_then_exactly_one_selected(tmp_path):
    row, subset=auth(tmp_path)
    assert row["chunk_id"]==PINNED_ROW["chunk_id"] and subset["row_count"]==1
    assert subset["row"]==row_identity(rows()[3],3)

def test_functional_selector_uses_frozen_plan_without_crypto(tmp_path):
    selected,subset=authorize_functional_single_row(
      rows=functional_rows(),full_manifest_sha256=MANIFEST,
      eligibility_sha256=ELIG,ordered_identity_sha256=ORDER,
      synthesis_seed=0,run_id=RUN,output_dir=tmp_path/"out",
      expected_chunk_id=FUNCTIONAL_ROW_ID,
      expected_chunk_sha256=FUNCTIONAL_ROW_SHA)
    assert selected["chunk_id"]==FUNCTIONAL_ROW_ID
    assert subset["row_count"]==1 and subset["synthesis_seed"]==0
    assert not any(key in subset for key in ("signature","ticket","public_key"))

def test_functional_selector_rejects_seed_and_ambiguous_row(tmp_path):
    kwargs=dict(full_manifest_sha256=MANIFEST,eligibility_sha256=ELIG,
      ordered_identity_sha256=ORDER,run_id=RUN,output_dir=tmp_path/"out",
      expected_chunk_id=FUNCTIONAL_ROW_ID,
      expected_chunk_sha256=FUNCTIONAL_ROW_SHA)
    with pytest.raises(ValueError,match="seed 0"):
      authorize_functional_single_row(
        rows=functional_rows(),synthesis_seed=1,**kwargs)
    duplicate=functional_rows(); duplicate[4]=copy.deepcopy(duplicate[3])
    with pytest.raises(ValueError,match="ambiguous"):
      authorize_functional_single_row(rows=duplicate,synthesis_seed=0,**kwargs)

def test_functional_selector_rejects_ineligible_preference_before_dispatch(
    tmp_path,
):
    cohort=functional_rows()
    row=cohort[3]
    row["chunk_id"]=PINNED_ROW["chunk_id"]
    row["chunk_sha256"]=PINNED_ROW["chunk_sha256"]
    row["_chunk"]={
      "id":PINNED_ROW["chunk_id"],
      "text":(
        "Two equations with equal slopes and equal intercepts describe the "
        "same line and therefore have infinitely many shared solutions."
      ),
      "learning_outcome_refs":["co-1"],
      "synthesis_focus_objective":row["focus_objective"],
    }
    with pytest.raises(ValueError,match="not DPO eligible"):
      authorize_functional_single_row(
        rows=cohort,full_manifest_sha256=MANIFEST,
        eligibility_sha256=ELIG,ordered_identity_sha256=ORDER,
        synthesis_seed=0,run_id=RUN,output_dir=tmp_path/"out",
        expected_chunk_id=PINNED_ROW["chunk_id"],
        expected_chunk_sha256=PINNED_ROW["chunk_sha256"])

def test_functional_preflight_persists_and_reobserves_raw_sources(
    tmp_path, monkeypatch,
):
    config={"served_model":"super","model_revision":"rev","backend":"trtllm",
      "backend_version":"1","max_context_tokens":32768,
      "max_output_tokens":2048,"strict_dialect":"openai_json_schema_strict",
      "thinking_enabled":False,"health":"ready","capacity_available":True,
      "active_clients":0,"workflow_paused":True,"stop_sentinel_clear":True,
      "tokenizer_identity":"tok"}
    cp=tmp_path/"config.json"; cp.write_text(json.dumps(config))
    sp=tmp_path/"schema.json"; sp.write_bytes(
      Path("schemas/knowledge/preference_pair.schema.json").read_bytes())
    raw=json.dumps({"data":[{"id":"super"}]}).encode()
    class Response:
      status=200
      def read(self): return raw
      def __enter__(self): return self
      def __exit__(self,*args): return False
    monkeypatch.setattr("urllib.request.urlopen",lambda *a,**k:Response())
    report=collect_functional_preflight(endpoint="http://control/v1",
      backend_config_path=cp,schema_path=sp,output_dir=tmp_path/"evidence",
      expected_model="super")
    assert report["generation_requests"]==0
    assert set(report["raw_sources"])=={"models_first_sha256",
      "models_second_sha256","backend_config_sha256",
      "projection_schema_sha256"}
    assert len(report["schema_registry_sha256"])==64

def test_functional_postflight_is_new_observation_and_reasoning_is_measured(
    tmp_path, monkeypatch,
):
    config={"served_model":"super","model_revision":"rev","backend":"trtllm",
      "backend_version":"1","max_context_tokens":32768,
      "max_output_tokens":2048,"strict_dialect":"openai_json_schema_strict",
      "thinking_enabled":False,"health":"ready","capacity_available":True,
      "active_clients":0,"workflow_paused":True,"stop_sentinel_clear":True,
      "tokenizer_identity":"tok"}
    cp=tmp_path/"config.json"; cp.write_text(json.dumps(config))
    raw=b'{"data":[{"id":"super"}]}'
    class Response:
      status=200
      def read(self): return raw
      def __enter__(self): return self
      def __exit__(self,*args): return False
    monkeypatch.setattr("urllib.request.urlopen",lambda *a,**k:Response())
    monkeypatch.setattr("time.monotonic",lambda:20.0)
    report=collect_functional_postflight(endpoint="http://control/v1",
      backend_config_path=cp,preflight=config,output_dir=tmp_path/"post",
      last_terminal_monotonic_seconds=10.0)
    assert report["active_clients"]==0
    assert report["observed_monotonic_seconds"]>10.0
    assert functional_reasoning_bytes([
      b'{"choices":[{"message":{"reasoning_content":"abc"}}]}'
    ])==3

@pytest.mark.parametrize("field,value",[
 ("synthesis_seed",1),("run_id","wrong"),("output_dir",Path("/wrong")),
 ("full_manifest_sha256","b"*64),("eligibility_sha256","b"*64),
 ("ordered_identity_sha256","b"*64)])
def test_wrong_go_identity_fails(tmp_path,field,value):
    with pytest.raises(ValueError):
        auth(tmp_path,mutate=lambda d:d.__setitem__(field,value))

def test_wrong_row_and_cohort_cardinality_fail(tmp_path):
    bad=rows(); bad[3]["chunk_id"]="x"
    with pytest.raises(ValueError,match="authorized row"):
        auth(tmp_path,rs=bad)
    for count in (2,7):
        with pytest.raises(ValueError,match="frozen8"):
            auth(tmp_path/str(count),rs=rows()[:count])
    with pytest.raises(ValueError,match="frozen8"):
        auth(tmp_path/"9",rs=rows()+[copy.deepcopy(rows()[0])])

def test_bad_signature_fails(tmp_path):
    go,p,cap,trust,authority=signed(tmp_path)
    d=json.loads(p.read_text()); d["payload"]["tuple_sha256"]="0"*64; p.write_text(json.dumps(d))
    with pytest.raises(Exception):
        authorize_single_row(rows=rows(),full_manifest_sha256=MANIFEST,
          eligibility_sha256=ELIG,ordered_identity_sha256=ORDER,synthesis_seed=0,
          run_id=RUN,output_dir=OUT,go_path=go,wrapper_path=p,
          capability_path=cap,trust_root_path=trust,
          gate_a_authority_path=authority)

def test_call_budget_order_dialect_repairs_extra_b_and_eighth(tmp_path):
    state=tmp_path/"state.json"; write_unconsumed(state,{"subset_sha256":H})
    c=GateDCallController(state_path=state,binding={"subset_sha256":H})
    for stage in ("micro_A_task","micro_B_claim_0","micro_B_claim_1",
                  "micro_B_claim_2","micro_D_realize","micro_E_bad","micro_F_final"):
        c.before_request(stage)
    with pytest.raises(RuntimeError,match="eighth"): c.before_request("micro_F_more")
    for stage,match in (("micro_dialect_preflight","dialect"),
                        ("micro_C_zero","unknown")):
        with pytest.raises(RuntimeError,match=match):
            GateDCallController(state_path=tmp_path/f"{match}.json",
             binding={}).before_request(stage)
    b=GateDCallController(state_path=tmp_path/"b.json",binding={})
    for i in range(3): b.before_request(f"micro_B_slot_{i}")
    with pytest.raises(RuntimeError,match="extra B"):
        b.before_request("micro_B_slot_3")

def test_repair_attempt_rejected_before_provider_dispatch(tmp_path):
    class P:
        calls=0
        def _call_stage(self,**kw): self.calls+=1
    state=tmp_path/"state"; write_unconsumed(state,{})
    p=P(); GateDCallController(state_path=state,binding={}).wrap(p)
    with pytest.raises(RuntimeError,match="repairs"):
        p._call_stage(stage="micro_A_task",max_stage_repairs=1)
    assert p.calls==0 and json.loads(state.read_text())["state"]=="unconsumed"

def test_started_consumption_terminal_and_reuse_prevention(tmp_path):
    state=tmp_path/"state"; write_unconsumed(state,{"go":H})
    c=GateDCallController(state_path=state,binding={"go":H})
    c.before_request("micro_A_task")
    c.http_attempt_started()
    assert json.loads(state.read_text())["state"]=="started"
    with pytest.raises(RuntimeError,match="consumed"):
        GateDCallController(state_path=state,binding={"go":H})
    c.terminal(outcome="failed")
    assert json.loads(state.read_text())["state"]=="terminal"

def test_completed_requires_exact_a_b_d_e_f_traversal(tmp_path):
    state=tmp_path/"state"; write_unconsumed(state,{})
    c=GateDCallController(state_path=state,binding={})
    for stage in ("micro_A_task","micro_B_claim","micro_D_realize",
                  "micro_E_reject","micro_F_finalize"):
        c.before_request(stage)
    c.http_attempt_started()
    c.terminal(outcome="completed")
    assert json.loads(state.read_text())["outcome"]=="completed"
    bad=tmp_path/"bad"; write_unconsumed(bad,{})
    c=GateDCallController(state_path=bad,binding={})
    c.before_request("micro_A_task")
    c.http_attempt_started()
    c.terminal(outcome="completed")
    assert json.loads(bad.read_text())["outcome"]=="failed_incomplete_traversal"

def test_zero_call_failure_remains_unconsumed(tmp_path):
    state=tmp_path/"state"; write_unconsumed(state,{"go":H})
    c=GateDCallController(state_path=state,binding={"go":H})
    c.terminal(outcome="preflight_failed")
    assert json.loads(state.read_text())["state"]=="unconsumed"

def test_operator_trust_root_fingerprint_is_not_self_selected(tmp_path):
    with pytest.raises(ValueError,match="fingerprint"):
        auth(tmp_path,mutate=lambda d:d["trust_root_path"].write_bytes(b"x"*32))

def test_symlink_authority_is_rejected(tmp_path):
    go,wp,cap,trust,fp=signed(tmp_path)
    link=tmp_path/"linked-go"; link.symlink_to(go)
    with pytest.raises(OSError):
        auth(tmp_path/"next",mutate=lambda d:d.__setitem__("go_path",link))

def test_fake_capability_or_thinking_on_is_rejected(tmp_path):
    def corrupt(values):
        cap=json.loads(values["capability_path"].read_text())
        cap["proof"]["thinking_enabled"]=True
        values["capability_path"].write_text(json.dumps(cap))
    with pytest.raises(ValueError):
        auth(tmp_path,mutate=corrupt)

def test_exact_schedule_rejects_b_before_a_skipped_slot_and_attempt2(tmp_path):
    expected=["micro_A_task_design","micro_B_claim_0_attempt_1",
      "micro_B_claim_1_attempt_1","micro_D_dpo_chosen",
      "micro_E_misconception_selection","micro_F_one_fault_rejected"]
    for first in ("micro_B_claim_0_attempt_1","micro_A_task_design"):
        c=GateDCallController(state_path=tmp_path/f"{len(first)}.json",
          binding={},expected_stages=expected)
        if first.startswith("micro_B"):
            with pytest.raises(RuntimeError,match="schedule"): c.before_request(first)
        else:
            c.before_request(first)
            with pytest.raises(RuntimeError,match="schedule"):
                c.before_request("micro_B_claim_1_attempt_1")
    c=GateDCallController(state_path=tmp_path/"attempt.json",binding={},
      expected_stages=expected)
    c.before_request(expected[0])
    with pytest.raises(RuntimeError,match="schedule"):
        c.before_request("micro_B_claim_0_attempt_2")

def test_functional_production_policy_requires_evidence_after_provider_traversal(
    tmp_path,
):
    state=tmp_path/"state.json"; write_unconsumed(state,{"subset_sha256":H})
    planned=["micro_A_task_design","micro_B_claim_0_attempt_1",
      "micro_D_dpo_chosen","micro_E_misconception_selection",
      "micro_F_one_fault_rejected"]
    c=GateDCallController(state_path=state,binding={"subset_sha256":H},
      expected_stages=planned,production_repairs=True,max_calls=50)
    for stage in ("micro_A_task_design","micro_A_task_design",
      "micro_B_claim_0_attempt_1","micro_D_dpo_chosen",
      "micro_E_misconception_selection","micro_F_one_fault_rejected"):
      c.before_request(stage)
    c._consumed_here=True
    assert c.terminal(outcome="completed")=="failed_incomplete_traversal"

def test_functional_production_policy_enforces_ceiling_and_frozen_slots(
    tmp_path,
):
    state=tmp_path/"state.json"; write_unconsumed(state,{})
    planned=["micro_A_task_design","micro_B_claim_0_attempt_1",
      "micro_D_dpo_chosen","micro_E_misconception_selection",
      "micro_F_one_fault_rejected"]
    c=GateDCallController(state_path=state,binding={},
      expected_stages=planned,production_repairs=True,max_calls=2)
    with pytest.raises(RuntimeError,match="outside frozen schedule"):
      c.before_request("micro_B_claim_1_attempt_1")
    class Stage:
      def get(self): return "micro_A_task_design"
    class Ledger:
      _stage=Stage()
      def record_started(self,**kwargs): return None
    ledger=Ledger(); c.install_http_attempt_hook(ledger)
    ledger.record_started(); ledger.record_started()
    with pytest.raises(RuntimeError,match="ceiling"):
      ledger.record_started()

def test_concurrent_consumers_only_one_cas_wins(tmp_path):
    state=tmp_path/"state"; write_unconsumed(state,{})
    controllers=[GateDCallController(state_path=state,binding={}) for _ in range(2)]
    outcomes=[]
    def consume(c):
        try: c.http_attempt_started(); outcomes.append("ok")
        except RuntimeError: outcomes.append("rejected")
    threads=[threading.Thread(target=consume,args=(c,)) for c in controllers]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert sorted(outcomes)==["ok","rejected"]

def test_transactional_precommit_tamper_fails(tmp_path):
    state=tmp_path/"state.json"; write_unconsumed(state,{})
    c=GateDCallController(state_path=state,binding={})
    c.before_request("micro_A_task"); c.http_attempt_started()
    c.terminal(outcome="failed")
    candidate={"schema":"ed4all.gate-d-precommit-candidate.v1",
      "binding_sha256":H,"policy_sha256":H,"schedule_sha256":H,
      "results_sha256":H,"summary_sha256":H,
      "consumption_sha256":__import__("hashlib").sha256(state.read_bytes()).hexdigest(),
      "schema_registry_sha256":_offline_pair_validator(
        Path("schemas/knowledge/preference_pair.schema.json"))[1]["sha256"]}
    path=tmp_path/"candidate.json"; path.write_text(json.dumps(candidate)); path.chmod(0o600)
    with pytest.raises(ValueError,match="consumption"):
        verify_gate_d_precommit(path,expected_binding_sha256=H,
          consumption_path=state)

def test_secure_output_tree_retains_dirfd_and_rejects_symlink(tmp_path):
    trusted=tmp_path/"trusted"; trusted.mkdir(mode=0o700)
    tree=SecureOutputTree(trusted_root=trusted,output_dir=trusted/"run")
    tree.create("proof.json",b"{}")
    assert tree.reopen("proof.json")[0]==b"{}"
    (trusted/"link").symlink_to(trusted/"run",target_is_directory=True)
    with pytest.raises(OSError):
        SecureOutputTree(trusted_root=trusted,output_dir=trusted/"link"/"x")
    tree.assert_identity(); tree.close()

def test_secure_output_tree_rejects_parent_substitution_race(tmp_path):
    trusted=tmp_path/"trusted"; trusted.mkdir(mode=0o700)
    tree=SecureOutputTree(trusted_root=trusted,output_dir=trusted/"run")
    (trusted/"run").rename(trusted/"displaced")
    (trusted/"run").mkdir(mode=0o700)
    with pytest.raises(RuntimeError,match="substituted"):
        tree.create("must-not-land",b"x")
    assert not (trusted/"run"/"must-not-land").exists()
    tree.close()

def test_secure_output_tree_rejects_existing_transaction_leaf(tmp_path):
    trusted=tmp_path/"trusted"; trusted.mkdir(mode=0o700)
    (trusted/"run").mkdir(mode=0o700)
    with pytest.raises(FileExistsError):
        SecureOutputTree(trusted_root=trusted,output_dir=trusted/"run")

def _transaction_fixture(tmp_path):
    root=tmp_path/"tx"; root.mkdir(mode=0o700)
    stages=["micro_A_task_design","micro_B_claim_0_attempt_1",
      "micro_D_dpo_chosen","micro_E_misconception_selection",
      "micro_F_one_fault_rejected"]
    def put(rel,obj,lines=False):
        p=root/rel; p.parent.mkdir(parents=True,exist_ok=True)
        raw=(("\n".join(json.dumps(x) for x in obj)+"\n") if lines
             else json.dumps(obj)).encode()
        p.write_bytes(raw); p.chmod(0o600); return p,raw
    schedule={"stages":[{"stage":s,"model_call":True} for s in stages]}
    sp,sraw=put("gate-d-request-schedule.json",schedule)
    state={"state":"terminal","outcome":"completed",
      "binding":{"subset_sha256":H}}
    cp,craw=put("gate-d-consumption.json",state)
    result=[{"accepted":True,
      "prompt":"Explain how grounded evidence supports a reliable answer.",
      "chosen":"A reliable answer explicitly connects every material claim to the supplied evidence and avoids unsupported additions.",
      "rejected":"A reliable answer can introduce plausible details that are absent from the evidence whenever they make the response clearer.",
      "chunk_id":"c",
      "lo_refs":["co-1"],"seed":0,"decision_capture_id":"dc-1",
      "claim_support_rate":1.0,"claim_contradicted_rate":0.0,
      "bloom_alignment":True,"promotion_status":"validated"}]
    rp,rraw=put("results.jsonl",result,lines=True)
    sump,sumraw=put("summary.json",{"gate_d_binding":{"subset_sha256":H}})
    intents=[{"stage":s} for s in stages]
    put("call-intents.jsonl",intents,lines=True)
    http=[]
    for index,s in enumerate(stages):
        _,request_raw=put(f"audit/http-raw/request-{index}.json",{"stage":s})
        _,response_raw=put(f"audit/http-raw/response-{index}.json",{"ok":True})
        request_sha=__import__("hashlib").sha256(request_raw).hexdigest()
        response_sha=__import__("hashlib").sha256(response_raw).hexdigest()
        http += [{"event":"http_attempt_started","stage":s,"attempt":1,
          "request_sha256":request_sha,"request_raw_ref":{"sha256":request_sha}},
          {"event":"http_attempt_terminal","stage":s,"attempt":1,
           "finish_reason":"stop","response_raw_ref":{"sha256":response_sha}}]
    put("http_attempts.jsonl",http,lines=True)
    put("run.checkpoint.jsonl",[{"state":"terminal","subset_sha256":H}],lines=True)
    put("micro-journals/unit.jsonl",[
      {"state":"terminal","stage":stage,"subset_sha256":H}
      for stage in ("A","B","C","D","E","F")],lines=True)
    put("audit/decision-capture/events.jsonl",
        [{"decision_type":"synthesis_provider_call","subset_sha256":H}
         for _ in stages],lines=True)
    put("telemetry/summary.json",{"abort_disconnect_count":0,
      "sampler_state":{"errors":[]}})
    candidate={"schema":"ed4all.gate-d-precommit-candidate.v1",
      "binding_sha256":H,"policy_sha256":H,
      "schedule_sha256":__import__("hashlib").sha256(sraw).hexdigest(),
      "results_sha256":__import__("hashlib").sha256(rraw).hexdigest(),
      "summary_sha256":__import__("hashlib").sha256(sumraw).hexdigest(),
      "consumption_sha256":__import__("hashlib").sha256(craw).hexdigest(),
      "schema_registry_sha256":_offline_pair_validator(
        Path("schemas/knowledge/preference_pair.schema.json"))[1]["sha256"]}
    cand,_=put("gate-d-precommit-candidate.json",candidate)
    return root,cand,stages

def _functional_transaction_fixture(tmp_path):
    root,candidate,stages=_transaction_fixture(tmp_path)
    stages=[stage for stage in stages if not stage.startswith("micro_A_")]
    preflight_raw=root/"functional-preflight/raw"
    preflight_raw.mkdir(parents=True)
    schema_bytes=Path("schemas/knowledge/preference_pair.schema.json").read_bytes()
    backend_config={"served_model":"super","model_revision":"rev",
      "backend":"trtllm","backend_version":"1","max_context_tokens":32768,
      "max_output_tokens":2048,"strict_dialect":"openai_json_schema_strict",
      "thinking_enabled":False,"health":"ready","capacity_available":True,
      "active_clients":0,"workflow_paused":True,"stop_sentinel_clear":True,
      "tokenizer_identity":"tok"}
    raw_values={
      "models-first.bin":b'{"data":[{"id":"super"}]}',
      "models-second.bin":b'{"data":[{"id":"super"}]}',
      "backend-config.json":_stable(backend_config).encode(),
      "projection-schema.json":schema_bytes,
    }
    for name,raw in raw_values.items():
      path=preflight_raw/name; path.write_bytes(raw); path.chmod(0o600)
    preflight={"schema":"ed4all.gate-d-functional-preflight.v1",
      **backend_config,
      "generation_requests":0,
      "schema_sha256":__import__("hashlib").sha256(schema_bytes).hexdigest(),
      "schema_registry_sha256":_offline_pair_validator(
        Path("schemas/knowledge/preference_pair.schema.json"))[1]["sha256"],
      "raw_sources":{
        "models_first_sha256":__import__("hashlib").sha256(
          raw_values["models-first.bin"]).hexdigest(),
        "models_second_sha256":__import__("hashlib").sha256(
          raw_values["models-second.bin"]).hexdigest(),
        "backend_config_sha256":__import__("hashlib").sha256(
          raw_values["backend-config.json"]).hexdigest(),
        "projection_schema_sha256":__import__("hashlib").sha256(
          schema_bytes).hexdigest()}}
    preflight_path=root/"functional-preflight/preflight.json"
    preflight_path.write_text(json.dumps(preflight)); preflight_path.chmod(0o600)
    state_path=root/"gate-d-consumption.json"
    state=json.loads(state_path.read_text())
    state["binding"]["strict_dialect_capability_sha256"]=_sha(preflight)
    state_path.write_text(json.dumps(state)); state_path.chmod(0o600)
    result=json.loads((root/"results.jsonl").read_text().strip())
    result.update({"projection_contract":"ed4all-dpo-preference.v2",
      "schema_version":"v1"})
    (root/"results.jsonl").write_text(_stable(result)+"\n")
    policy={"version":"gate-d-single-pass.v1",
      "limits":{"claim_attempts_per_slot":1,"semantic_repairs_per_stage":0,
        "leakage_repairs_per_stage":0},"trusted_binding_sha256":H}
    policy["sha256"]=_sha(policy)
    schedule={"functional_version":"1.3.1","binding_sha256":H,
      "policy":policy,"stages":[
        {"stage":"micro_A_task_design","model_call":False},
        *[{"stage":s,"model_call":True} for s in stages
          if s.startswith("micro_B_")],
        {"stage":"micro_C_assembly","model_call":False},
        *[{"stage":s,"model_call":True} for s in stages
          if not s.startswith("micro_B_")],
      ]}
    (root/"gate-d-request-schedule.json").write_text(json.dumps(schedule))
    intents=[]
    http=[]
    for path in (root/"audit/http-raw").glob("*"):
      path.unlink()
    budgets={"A":2048,"B":1536,"D":1536,"E":1280,"F":1024}
    for index,stage in enumerate(stages):
      letter=stage.split("_")[1]
      payload={"model":"super","max_tokens":budgets[letter],
        "chat_template_kwargs":{"enable_thinking":False},
        "response_format":{"json_schema":{"schema":{"type":"object"}}}}
      request_raw=_stable(payload).encode()
      request_path=root/f"audit/http-raw/request-{index}.json"
      request_path.write_bytes(request_raw); request_path.chmod(0o600)
      response_raw=json.dumps({"choices":[{"finish_reason":"stop"}]}).encode()
      response_path=root/f"audit/http-raw/response-{index}.json"
      response_path.write_bytes(response_raw); response_path.chmod(0o600)
      request_sha=__import__("hashlib").sha256(request_raw).hexdigest()
      response_sha=__import__("hashlib").sha256(response_raw).hexdigest()
      schema_sha=_sha({"type":"object"})
      contract_sha=_sha({"stage":stage,"model":"super",
        "max_tokens":budgets[letter]})
      intents.append({"stage":f"staged_synthesis:{stage}",
        "logical_attempt":1,"kind":"initial",
        "request_sha256":request_sha,"model":"super",
        "model_revision":"rev",
        "max_tokens":budgets[letter],"response_schema_sha256":schema_sha,
        "contract_sha256":contract_sha})
      http.extend([
        {"event":"http_attempt_started","stage":stage,"attempt":1,
          "model":"super","model_revision":"rev",
          "endpoint":"http://provider.invalid/v1/chat/completions",
          "monotonic_seconds":100.0+index,
          "request_sha256":request_sha,"request_raw_ref":{"sha256":request_sha}},
        {"event":"http_attempt_terminal","stage":f"staged_synthesis:{stage}",
          "attempt":1,"request_sha256":request_sha,"finish_reason":"stop",
          "http_status":200,
          "model":"super","model_revision":"rev",
          "endpoint":"http://provider.invalid/v1/chat/completions",
          "monotonic_seconds":100.5+index,
          "exception_class":None,"response_raw_ref":{"sha256":response_sha}},
      ])
    def lines(path,values):
      path.write_text("".join(_stable(v)+"\n" for v in values)); path.chmod(0o600)
    lines(root/"call-intents.jsonl",intents)
    lines(root/"http_attempts.jsonl",http)
    decisions=[]
    dcraw=root/"audit/decision-capture/raw"; dcraw.mkdir(parents=True,exist_ok=True)
    for index,(stage,intent) in enumerate(zip(stages,intents)):
      prompt=f"prompt-{index}".encode(); response=f"response-{index}".encode()
      pp=dcraw/f"prompt-{index}"; rp=dcraw/f"response-{index}"
      pp.write_bytes(prompt); rp.write_bytes(response); pp.chmod(0o600); rp.chmod(0o600)
      decisions.append({"decision_type":"synthesis_provider_call",
        "rationale":f"dynamic stage {stage} exact model and token evidence",
        "context":_stable({"stage":stage,"attempt":1,
          "model":"super","model_revision":"rev",
          "intent_request_sha256":intent["request_sha256"],
          "intent_model":"super",
          "intent_contract_sha256":intent["contract_sha256"],
          "trusted_binding_sha256":H,
          "response_schema_sha256":intent["response_schema_sha256"],
          "prompt_sha256":__import__("hashlib").sha256(prompt).hexdigest(),
          "response_sha256":__import__("hashlib").sha256(response).hexdigest()})})
    lines(root/"audit/decision-capture/events.jsonl",decisions)
    journals=[]
    previous="0"*64
    sequence=0
    terminal_order=[("A",None),("B",0),("C",None),("D",None),("E",None),("F",None)]
    for letter,slot in terminal_order:
      for state in ("started","terminal"):
        sequence+=1
        row={"sequence":sequence,"previous_sha256":previous,
          "contract_fingerprint":H,
          "unit":f"preference:{letter}"+(f":slot-{slot}" if slot is not None else ""),
          "stage":letter,"slot":slot,"attempt":1,"state":state,
          "artifact":({
            "_stage_a_deterministic":{
              "model_calls":0,"decision_capture_events":0,
              "telemetry":{"deterministic_events":1,"model_calls":0,
                "prompt_tokens":0,"completion_tokens":0,"total_tokens":0}},
          } if letter=="A" else {
            "_stage_c_deterministic":{
              "model_calls":0,"decision_capture_events":0,
              "telemetry":{"deterministic_events":1,"model_calls":0,
                "prompt_tokens":0,"completion_tokens":0,"total_tokens":0}},
          } if letter=="C" else {}) if state=="terminal" else None,
          "gate_d_binding":{"subset_sha256":H}}
        row["row_sha256"]=_sha(row); previous=row["row_sha256"]; journals.append(row)
    lines(root/"micro-journals/unit.jsonl",journals)
    lines(root/"run.checkpoint.jsonl",[{"_checkpoint_state":"terminal",
      "accepted":True,"stage_validity":True,
      "gate_d_binding":{"subset_sha256":H},"result":result}])
    telemetry={"abort_disconnect_count":0,
      "sampler_state":{"errors":[],"stop_requested":True},
      "request_count":len(stages),"stage_request_counts":{s:1 for s in stages},
      "active_clients_final":0,"reasoning_bytes":0,
      "finish_reason_counts":{"stop":len(stages)},
      "postflight_observed_monotonic_seconds":200.0,
      "token_observations":[{
        "stage":stage,"prompt_tokens":4,"completion_tokens":6,
        "total_tokens":10,"max_output_tokens":budgets[stage.split("_")[1]],
        "output_headroom_tokens":budgets[stage.split("_")[1]]-6,
      } for stage in stages],
      "gpu_observations":[{"timestamp":"now","gpu_utilization_percent":1.0,
        "memory_utilization_percent":2.0,"power_watts":3.0,
        "temperature_c":40.0}],
      "kv_observations":[{"kv_blocks":1,"peak_scheduled_token_usage":10,
        "peak_scheduled_token_headroom":8182}],
      "raw_sampler_sources":{},"verifier_accepted":True}
    telemetry_root=root/"telemetry"
    (telemetry_root/"trtllm.log").write_text("scheduled tokens")
    (telemetry_root/"system.jsonl").write_text('{"gpu":["now,1,2,3,40"]}\n')
    for path in (telemetry_root/"trtllm.log",telemetry_root/"system.jsonl"):
      path.chmod(0o600)
      telemetry["raw_sampler_sources"][path.name]=__import__("hashlib").sha256(
        path.read_bytes()).hexdigest()
    (root/"telemetry/summary.json").write_text(json.dumps(telemetry))
    post_root=root/"functional-postflight/raw"; post_root.mkdir(parents=True)
    post_models=raw_values["models-first.bin"]
    post_config=raw_values["backend-config.json"]
    for name,raw in (("models.bin",post_models),
                     ("backend-config.json",post_config)):
      p=post_root/name; p.write_bytes(raw); p.chmod(0o600)
    postflight={"schema":"ed4all.gate-d-functional-postflight.v1",
      **backend_config,"endpoint":"http://control/v1",
      "models_json_sha256":_sha(json.loads(post_models)),
      "models_raw_sha256":__import__("hashlib").sha256(post_models).hexdigest(),
      "backend_config_sha256":__import__("hashlib").sha256(post_config).hexdigest(),
      "last_terminal_monotonic_seconds":100.5+len(stages)-1,
      "observed_monotonic_seconds":200.0,"observed_utc_ns":1,
      "generation_requests":0}
    post_path=root/"functional-postflight/postflight.json"
    post_path.write_text(json.dumps(postflight)); post_path.chmod(0o600)
    schedule_raw=(root/"gate-d-request-schedule.json").read_bytes()
    candidate_doc=json.loads(candidate.read_text())
    candidate_doc["policy_sha256"]=policy["sha256"]
    candidate_doc["schedule_sha256"]=__import__("hashlib").sha256(schedule_raw).hexdigest()
    candidate_doc["results_sha256"]=__import__("hashlib").sha256(
      (root/"results.jsonl").read_bytes()).hexdigest()
    candidate_doc["consumption_sha256"]=__import__("hashlib").sha256(
      state_path.read_bytes()).hexdigest()
    candidate.write_text(json.dumps(candidate_doc)); candidate.chmod(0o600)
    return root,candidate,stages

def test_functional_full_reconciliation_accepts_exact_transaction(tmp_path):
    root,candidate,stages=_functional_transaction_fixture(tmp_path)
    report=verify_full_gate_d_transaction(root,expected_stages=stages,
      expected_binding_sha256=H,candidate_path=candidate)
    assert report["verified"]

def _rewrite_functional_pair(root, candidate, mutate):
    result_path=root/"results.jsonl"
    pair=json.loads(result_path.read_text())
    mutate(pair)
    result_path.write_text(_stable(pair)+"\n")
    checkpoint_path=root/"run.checkpoint.jsonl"
    checkpoint=json.loads(checkpoint_path.read_text())
    checkpoint["result"]=pair
    checkpoint_path.write_text(_stable(checkpoint)+"\n")
    candidate_doc=json.loads(candidate.read_text())
    candidate_doc["results_sha256"]=__import__("hashlib").sha256(
      result_path.read_bytes()).hexdigest()
    candidate.write_text(json.dumps(candidate_doc))

def _stamp_complete_bloom_authority(pair):
    pair["bloom_alignment"]=None
    pair["bloom_level"]="analyze"
    pair["lo_refs"]=["co-1"]
    pair["pair_objective_alignment_pass_rate"]=1.0
    pair["pair_objective_alignment"]=[{
      "objective_id":"co-1","status":"delivered",
      "statement_entailment_score":0.78,"contradiction_score":0.10,
      "bloom_gap":0,"verb_match_count":1,"declared_bloom":"analyze",
      "observed_bloom":"analyze","entailment_threshold":0.45,
    }]

def test_functional_full_reconciliation_accepts_canary028_null_bloom_authority(
    tmp_path,
):
    root,candidate,stages=_functional_transaction_fixture(tmp_path)
    _rewrite_functional_pair(root,candidate,_stamp_complete_bloom_authority)
    report=verify_full_gate_d_transaction(root,expected_stages=stages,
      expected_binding_sha256=H,candidate_path=candidate)
    assert report["verified"]

@pytest.mark.parametrize("mutation",[
  "explicit_false","null_alignment","threshold","objective_id",
  "entailment","contradiction","observed","declared","gap","order",
])
def test_functional_null_bloom_authority_mutations_fail_closed(
    tmp_path, mutation,
):
    root,candidate,stages=_functional_transaction_fixture(tmp_path)
    def mutate(pair):
      _stamp_complete_bloom_authority(pair)
      entry=pair["pair_objective_alignment"][0]
      if mutation=="explicit_false": pair["bloom_alignment"]=False
      elif mutation=="null_alignment": pair["pair_objective_alignment"]=None
      elif mutation=="threshold": entry["entailment_threshold"]=0.01
      elif mutation=="objective_id": entry["objective_id"]="co-other"
      elif mutation=="entailment": entry["statement_entailment_score"]=0.44
      elif mutation=="contradiction": entry["contradiction_score"]=0.50
      elif mutation=="observed": entry["observed_bloom"]="apply"
      elif mutation=="declared": entry["declared_bloom"]="apply"
      elif mutation=="gap": entry["bloom_gap"]=1
      else:
        pair["lo_refs"]=["co-2","co-1"]
        pair["pair_objective_alignment"]=[dict(entry),dict(entry)]
        pair["pair_objective_alignment"][1]["objective_id"]="co-2"
    _rewrite_functional_pair(root,candidate,mutate)
    with pytest.raises(ValueError,match="semantic validators"):
      verify_full_gate_d_transaction(root,expected_stages=stages,
        expected_binding_sha256=H,candidate_path=candidate)

def _terminal_controller(root, stages):
    state=root/"gate-d-consumption.json"
    state.unlink()
    write_unconsumed(state,{"subset_sha256":H})
    controller=GateDCallController(
      state_path=state,binding={"subset_sha256":H},
      expected_stages=stages,production_repairs=True,max_calls=45)
    for stage in stages:
      controller.before_request(stage)
    controller.http_calls=[
      f"staged_synthesis:{stage}" for stage in stages
    ]
    starts=[
      json.loads(line) for line in
      (root/"http_attempts.jsonl").read_text().splitlines()
      if json.loads(line).get("event")=="http_attempt_started"
    ]
    controller.http_call_evidence=[{
      "stage":f"staged_synthesis:{stage}",
      "attempt":row["attempt"],"request_sha256":row["request_sha256"],
      "model":row["model"],"endpoint":row["endpoint"],
    } for stage,row in zip(stages,starts,strict=True)]
    controller._consumed_here=True
    return controller

def _rehash_journal(path):
    previous="0"*64
    rows_=[json.loads(line) for line in path.read_text().splitlines()]
    for sequence,row in enumerate(rows_,1):
      row["sequence"]=sequence
      row["previous_sha256"]=previous
      row.pop("row_sha256",None)
      row["row_sha256"]=__import__("hashlib").sha256(json.dumps(
        row,sort_keys=True,separators=(",",":"),ensure_ascii=False,
      ).encode("utf-8")).hexdigest()
      previous=row["row_sha256"]
    path.write_text("".join(_stable(row)+"\n" for row in rows_))

def test_functional_terminal_reconciliation_accepts_deterministic_ac_and_calls(
    tmp_path,
):
    root,_candidate,stages=_functional_transaction_fixture(tmp_path)
    controller=_terminal_controller(root,stages)
    assert controller.terminal(
      outcome="completed",evidence_root=root)=="completed"
    state=json.loads((root/"gate-d-consumption.json").read_text())
    assert state["calls_started"]==4
    assert state["deterministic_stages_completed"]==["A","C"]
    assert state["provider_stages_completed"]==stages

def test_functional_terminal_reconciliation_preserves_semantic_retry_parity(
    tmp_path,
):
    root,_candidate,stages=_functional_transaction_fixture(tmp_path)
    retry_stage=stages[0].replace("_attempt_1","_attempt_2")
    intents=[json.loads(line) for line in
             (root/"call-intents.jsonl").read_text().splitlines()]
    retry=copy.deepcopy(intents[0]); retry["stage"]=f"staged_synthesis:{retry_stage}"
    intents.insert(1,retry)
    (root/"call-intents.jsonl").write_text(
      "".join(_stable(row)+"\n" for row in intents))
    http=[json.loads(line) for line in
          (root/"http_attempts.jsonl").read_text().splitlines()]
    retry_http=[copy.deepcopy(row) for row in http[:2]]
    for row in retry_http: row["stage"]=f"staged_synthesis:{retry_stage}"
    http[2:2]=retry_http
    (root/"http_attempts.jsonl").write_text(
      "".join(_stable(row)+"\n" for row in http))
    decisions=[json.loads(line) for line in
      (root/"audit/decision-capture/events.jsonl").read_text().splitlines()]
    retry_dc=copy.deepcopy(decisions[0])
    context=json.loads(retry_dc["context"]); context["stage"]=retry_stage
    retry_dc["context"]=_stable(context); decisions.insert(1,retry_dc)
    (root/"audit/decision-capture/events.jsonl").write_text(
      "".join(_stable(row)+"\n" for row in decisions))
    state=root/"gate-d-consumption.json"; state.unlink()
    write_unconsumed(state,{"subset_sha256":H})
    controller=GateDCallController(
      state_path=state,binding={"subset_sha256":H},
      expected_stages=stages,production_repairs=True,max_calls=45)
    actual=[stages[0],retry_stage,*stages[1:]]
    for stage in actual: controller.before_request(stage)
    controller.http_calls=[
      f"staged_synthesis:{stage}" for stage in actual]
    starts=[row for row in http if row["event"]=="http_attempt_started"]
    controller.http_call_evidence=[{
      "stage":f"staged_synthesis:{stage}","attempt":row["attempt"],
      "request_sha256":row["request_sha256"],"model":row["model"],
      "endpoint":row["endpoint"],
    } for stage,row in zip(actual,starts,strict=True)]
    controller._consumed_here=True
    assert controller.terminal(
      outcome="completed",evidence_root=root)=="completed"
    assert json.loads((root/"gate-d-consumption.json").read_text())[
      "calls_started"]==5

def test_functional_terminal_reconciliation_uses_utf8_journal_canonicalization(
    tmp_path,
):
    root,_candidate,stages=_functional_transaction_fixture(tmp_path)
    journal=root/"micro-journals/unit.jsonl"
    rows_=[json.loads(line) for line in journal.read_text().splitlines()]
    terminal_a=next(
      row for row in rows_ if row["stage"]=="A" and row["state"]=="terminal")
    terminal_a["artifact"]["canonicalization_probe"]="distance—sum"
    journal.write_text("".join(_stable(row)+"\n" for row in rows_))
    _rehash_journal(journal)
    controller=_terminal_controller(root,stages)
    assert controller.terminal(
      outcome="completed",evidence_root=root)=="completed"

def test_functional_terminal_reconciliation_counts_internal_e_repairs_physical(
    tmp_path,
):
    root,_candidate,stages=_functional_transaction_fixture(tmp_path)
    controller=_terminal_controller(root,stages)
    e_stage="micro_E_misconception_selection"
    intents=[json.loads(line) for line in
             (root/"call-intents.jsonl").read_text().splitlines()]
    e_index=next(i for i,row in enumerate(intents) if e_stage in row["stage"])
    intents[e_index:e_index]=[
      copy.deepcopy(intents[e_index]) for _ in range(2)]
    for attempt,row in enumerate(intents[e_index:e_index+3],1):
      row["logical_attempt"]=attempt
      row["kind"]="initial" if attempt==1 else "repair"
    (root/"call-intents.jsonl").write_text(
      "".join(_stable(row)+"\n" for row in intents))
    http=[json.loads(line) for line in
          (root/"http_attempts.jsonl").read_text().splitlines()]
    pair_index=next(i for i in range(0,len(http),2)
      if e_stage in http[i]["stage"])
    pair=copy.deepcopy(http[pair_index:pair_index+2])
    http[pair_index:pair_index]=[
      copy.deepcopy(row) for _ in range(2) for row in pair]
    for attempt,row in enumerate(http[pair_index:pair_index+6],1):
      row["attempt"]=(attempt+1)//2
    (root/"http_attempts.jsonl").write_text(
      "".join(_stable(row)+"\n" for row in http))
    decisions=[json.loads(line) for line in
      (root/"audit/decision-capture/events.jsonl").read_text().splitlines()]
    dc_index=next(i for i,row in enumerate(decisions)
      if e_stage in json.loads(row["context"])["stage"])
    decisions[dc_index:dc_index]=[
      copy.deepcopy(decisions[dc_index]) for _ in range(2)]
    for attempt,row in enumerate(decisions[dc_index:dc_index+3],1):
      context=json.loads(row["context"]); context["attempt"]=attempt
      row["context"]=_stable(context)
    (root/"audit/decision-capture/events.jsonl").write_text(
      "".join(_stable(row)+"\n" for row in decisions))
    e_http=controller.http_calls.index(f"staged_synthesis:{e_stage}")
    controller.http_calls[e_http+1:e_http+1]=[
      f"staged_synthesis:{e_stage}"]*2
    e_hook=copy.deepcopy(controller.http_call_evidence[e_http])
    e_hook["attempt"]=2
    e_hook_3=copy.deepcopy(e_hook); e_hook_3["attempt"]=3
    controller.http_call_evidence[e_http+1:e_http+1]=[e_hook,e_hook_3]
    assert controller.terminal(
      outcome="completed",evidence_root=root)=="completed"
    assert json.loads((root/"gate-d-consumption.json").read_text())[
      "calls_started"]==6

@pytest.mark.parametrize("mutation",[
  "missing_a","duplicate_c","reordered_ac","a_intent","missing_provider",
  "extra_provider","reordered_provider",
])
def test_functional_terminal_reconciliation_fails_closed_on_evidence_drift(
    tmp_path, mutation,
):
    root,_candidate,stages=_functional_transaction_fixture(tmp_path)
    controller=_terminal_controller(root,stages)
    journal=root/"micro-journals/unit.jsonl"
    if mutation in {"missing_a","duplicate_c","reordered_ac"}:
      rows_=[json.loads(line) for line in journal.read_text().splitlines()]
      if mutation=="missing_a":
        rows_=[row for row in rows_ if row["stage"]!="A"]
      elif mutation=="duplicate_c":
        rows_.append(copy.deepcopy(next(
          row for row in rows_ if row["stage"]=="C" and row["state"]=="terminal"
        )))
      else:
        a=next(i for i,row in enumerate(rows_) if
          row["stage"]=="A" and row["state"]=="terminal")
        c=next(i for i,row in enumerate(rows_) if
          row["stage"]=="C" and row["state"]=="terminal")
        rows_[a],rows_[c]=rows_[c],rows_[a]
      journal.write_text("".join(_stable(row)+"\n" for row in rows_))
      _rehash_journal(journal)
    elif mutation=="a_intent":
      for path in (root/"call-intents.jsonl",
                   root/"audit/decision-capture/events.jsonl"):
        rows_=[json.loads(line) for line in path.read_text().splitlines()]
        rows_.insert(0,copy.deepcopy(rows_[0]))
        if path.name=="call-intents.jsonl":
          rows_[0]["stage"]="staged_synthesis:micro_A_task_design"
        else:
          context=json.loads(rows_[0]["context"])
          context["stage"]="micro_A_task_design"
          rows_[0]["context"]=_stable(context)
        path.write_text("".join(_stable(row)+"\n" for row in rows_))
      rows_=[json.loads(line) for line in
             (root/"http_attempts.jsonl").read_text().splitlines()]
      extra=[copy.deepcopy(row) for row in rows_[:2]]
      for row in extra: row["stage"]="micro_A_task_design"
      (root/"http_attempts.jsonl").write_text(
        "".join(_stable(row)+"\n" for row in extra+rows_))
      controller.calls.insert(0,"A")
      controller.http_calls.insert(0,"staged_synthesis:micro_A_task_design")
    else:
      indices={"missing_provider":-1,"extra_provider":0,
               "reordered_provider":None}
      for path in (root/"call-intents.jsonl",
                   root/"audit/decision-capture/events.jsonl"):
        rows_=[json.loads(line) for line in path.read_text().splitlines()]
        if mutation=="missing_provider": rows_.pop()
        elif mutation=="extra_provider": rows_.append(copy.deepcopy(rows_[0]))
        else: rows_[0],rows_[1]=rows_[1],rows_[0]
        path.write_text("".join(_stable(row)+"\n" for row in rows_))
      rows_=[json.loads(line) for line in
             (root/"http_attempts.jsonl").read_text().splitlines()]
      pairs=[rows_[i:i+2] for i in range(0,len(rows_),2)]
      if mutation=="missing_provider": pairs.pop()
      elif mutation=="extra_provider": pairs.append(copy.deepcopy(pairs[0]))
      else: pairs[0],pairs[1]=pairs[1],pairs[0]
      (root/"http_attempts.jsonl").write_text(
        "".join(_stable(row)+"\n" for pair in pairs for row in pair))
      if mutation=="missing_provider":
        controller.calls.pop(); controller.http_calls.pop()
      elif mutation=="extra_provider":
        controller.calls.append(controller.calls[0])
        controller.http_calls.append(controller.http_calls[0])
      else:
        controller.calls[0],controller.calls[1]=(
          controller.calls[1],controller.calls[0])
        controller.http_calls[0],controller.http_calls[1]=(
          controller.http_calls[1],controller.http_calls[0])
    assert controller.terminal(
      outcome="completed",evidence_root=root)=="failed_incomplete_traversal"
    state=json.loads((root/"gate-d-consumption.json").read_text())
    assert state["outcome"]=="failed_incomplete_traversal"
    with pytest.raises(ValueError,match="consumption"):
      verify_gate_d_precommit(
        root/"gate-d-precommit-candidate.json",
        expected_binding_sha256=H,
        consumption_path=root/"gate-d-consumption.json",
      )

@pytest.mark.parametrize("mutation",[
  "delete_hook","duplicate_hook","reorder_hook","attempt","kind",
  "request_hash","model","endpoint",
])
def test_functional_terminal_positional_tuple_mutations_fail_closed(
    tmp_path, mutation,
):
    root,_candidate,stages=_functional_transaction_fixture(tmp_path)
    controller=_terminal_controller(root,stages)
    if mutation=="delete_hook":
      controller.http_call_evidence.pop()
    elif mutation=="duplicate_hook":
      controller.http_call_evidence.append(
        copy.deepcopy(controller.http_call_evidence[0]))
    elif mutation=="reorder_hook":
      controller.http_call_evidence[0],controller.http_call_evidence[1]=(
        controller.http_call_evidence[1],controller.http_call_evidence[0])
    elif mutation=="endpoint":
      controller.http_call_evidence[0]["endpoint"]="http://wrong.invalid/v1"
    elif mutation in {"attempt","kind","request_hash"}:
      path=root/"call-intents.jsonl"
      rows_=[json.loads(line) for line in path.read_text().splitlines()]
      key={"attempt":"logical_attempt","kind":"kind",
           "request_hash":"request_sha256"}[mutation]
      rows_[0][key]={"attempt":2,"kind":"repair",
                     "request_hash":"f"*64}[mutation]
      path.write_text("".join(_stable(row)+"\n" for row in rows_))
    else:
      path=root/"http_attempts.jsonl"
      rows_=[json.loads(line) for line in path.read_text().splitlines()]
      rows_[1]["model"]="wrong-model"
      path.write_text("".join(_stable(row)+"\n" for row in rows_))
    assert controller.terminal(
      outcome="completed",evidence_root=root)=="failed_incomplete_traversal"
    assert not (root/"gate-d-publication.json").exists()

def test_execute_pilot_real_seam_projects_complete_dpo_and_verifies(
    tmp_path, monkeypatch,
):
    from Trainforge.scripts.harness.staged_window_abcd_pilot import execute_pilot
    from Trainforge.generators.staged_synthesis_micro import (
      MicroStagedSynthesisProvider, micro_contract_fingerprint,
    )
    assert micro_contract_fingerprint() == (
        # Re-sealed for the Bloom-ladder wave: micro_preference_eligibility's
        # Arm A recompute-and-reject id check now uses each CARD's own
        # bloom_level (TRAINFORGE_BLOOM_WINDOWS per-card rung recovery)
        # instead of the single chunk-level value, and the evidence-window
        # call sites thread target_rung. Prior seal: Arm B defer-to-Arm-A on
        # a doubly-recovered authored card (a74f8959...).
        "50e14774ccc5731b7e8d64eb6dddf76ccd99c34ea81b8f6f318ab4d2ee544575"
    )
    root,candidate,stages=_functional_transaction_fixture(tmp_path)
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS","32768")
    class Capture:
      def __init__(self): self.decisions=[]
      def log_decision(self,**kwargs):
        self.decisions.append({"event_id":f"EVT_mock_{len(self.decisions)+1:04d}",
          **kwargs})
    class Score:
      entailment=0.99
      contradiction=0.0
    class Nli:
      @staticmethod
      def score_pair(**kwargs): return Score()
    class Intent:
      def __init__(self): self.active=None
      def admit(self,digest):
        self.active={"run_id":"run","cell_id":"cell","unit":"unit",
          "kind":"initial","stage":"strict","logical_attempt":1,
          "request_sha256":digest,"model":"super","contract_sha256":"c"*64}
      def current(self): return self.active
    class Client:
      @staticmethod
      def _extract_json_lenient(raw): return json.loads(raw)
    class Base:
      api_url="http://test.invalid/v1/chat/completions"
      base_url="http://test.invalid/v1"
      client=object()
      _oa_client=Client()
      _model="super"
      _provider_name="local"
      _provenance_provider="local"
      _max_tokens=2048
      _plan_nli_scorer=Nli()
      def __init__(self,responses):
        self.responses=list(responses); self.response_index=0
        self.schemas=[]
        self._capture=Capture()
        self.manifest=Intent()
      def _chat_completion_raw_structured(self,messages,*,schema,max_tokens):
        rendered=json.dumps(messages,sort_keys=True)
        self.manifest.admit(__import__("hashlib").sha256(rendered.encode()).hexdigest())
        self.schemas.append(schema)
        if self.response_index >= len(self.responses):
          raise AssertionError(
            f"scripted response schedule exhausted at call "
            f"{self.response_index + 1}; expected {len(self.responses)} calls"
          )
        response=self.responses[self.response_index]
        self.response_index += 1
        return response,{"prompt_tokens":13,"completion_tokens":8},0
    chunk={"id":"c","text":(
      "A transaction groups operations into one logical unit. Atomicity means "
      "operations all succeed or all fail, preventing partial updates. A "
      "mistaken view says atomic transactions preserve successful partial work."),
      "learning_outcome_refs":["co-1"],"bloom_level":"analyze",
      "misconceptions":[{"misconception":
        "Atomic transactions preserve successful partial work.",
        "correction":"Atomicity prevents partial updates."}],
      "synthesis_focus_objective":{"id":"co-1","statement":
        "Analyze how atomicity prevents partial updates.","bloom_level":"analyze",
        "bloom_verb":"analyze",
        "action_object":"how atomicity prevents partial updates",
        "conditions":[],"content_obligations":[
          "how atomicity prevents partial updates"],
        "performance_criteria":[],"content_obligation_anchors":[]}}
    claim={"claim":"Atomicity rules out leaving a transaction partly applied.",
      "evidence_quote":"Atomicity means operations all succeed or all fail, preventing partial updates.",
      "source_block_id":"c"}
    claim_id=_sha({**claim,"source_role":"flat_source",
      "source_polarity":"factual"})[:16]
    from Trainforge.generators.objective_execution_contract import (
      derive_objective_requirements,
    )
    requirement_contract=derive_objective_requirements(
      chunk["synthesis_focus_objective"])
    worked=[item for item in requirement_contract["requirements"]
      if item["kind"]!="result"]
    result_requirement=next(item for item in requirement_contract["requirements"]
      if item["kind"]=="result")
    misconception_id=canonical_mc_id(
      "Atomic transactions preserve successful partial work.",
      "Atomicity prevents partial updates.",
      "analyze",
    )
    responses=[
      json.dumps(claim),
      json.dumps({"claim_realizations":[{"claim_id":claim_id,
          "realization":"Atomicity rules out leaving a transaction partly applied."}],
        "worked_steps":[{"requirement_id":item["requirement_id"],
          "claim_ids":[claim_id],"realization":
          "Analyze the all-or-none relationship: atomicity rules out leaving a transaction partly applied."}
          for item in worked],
        "result":{"requirement_id":result_requirement["requirement_id"],
          "claim_ids":[claim_id],"realization":
          "Therefore, a transaction cannot remain partly applied."}}),
      json.dumps({"misconception_id":misconception_id,
        "rationale":"The selected misconception reverses the all-or-none transaction guarantee."}),
      json.dumps({"rejected":"Earlier successful operations remain applied even when a later operation fails, because partial success is preserved."}),
    ]
    base=Base(responses)
    micro=MicroStagedSynthesisProvider(base,synthesis_seed=0)
    micro._pilot_attempt_ledger=type("Ledger",(),{"_intent_manifest":base.manifest})()
    class Provider:
      _pilot_calls=[]
      _pilot_gate_d_binding={"subset_sha256":H}
      def paraphrase_preference(self,draft,chunk):
        return micro.paraphrase_preference(draft,chunk)
    source={"chunk_id":"c","chunk_sha256":"1"*64,"kind":"preference",
      "variant":"D_production_contract","repetition":0,
      "focus_objective":{"id":"co-1"},"_chunk":chunk}
    checkpoint=root/"run.checkpoint.jsonl"; checkpoint.unlink()
    def scorer(pair,*args):
      pair.update({"claim_support_rate":1.0,"claim_contradicted_rate":0.0,
        "bloom_alignment":True,"promotion_status":"validated"})
      return {"accepted":True}
    results,_=execute_pilot([source],Provider(),objectives=[],
      scorer=scorer,checkpoint_path=checkpoint)
    checkpoint.chmod(0o600)
    assert results[0]["result"] is not None, {
      "result":results[0],"responses_consumed":base.response_index,
      "schemas":[schema.get("required") for schema in base.schemas],
      "decisions":base._capture.decisions}
    assert set(results[0]["result"]) >= {"prompt","chosen","rejected"}
    assert results[0]["result"]["prompt"].startswith(
      "Analyze how atomicity prevents partial updates.")
    assert results[0]["result"]["prompt"].endswith(
      "do not copy an answer.")
    assert results[0]["result"]["projection_contract"] == (
      "ed4all-dpo-preference.v2")
    assert results[0]["result"]["provenance"]["claim_realizations"] == {
      claim_id:"Atomicity rules out leaving a transaction partly applied."}
    assert results[0]["result"]["provenance"]["assembled_realization"][
      "ordered_claim_ids"] == [claim_id]
    assert results[0]["result"]["misconception_id"] == misconception_id
    assert base.response_index==len(responses)==4
    assert [schema["required"] for schema in base.schemas]==[
        ["claim","evidence_quote","source_block_id"],
        ["claim_realizations","worked_steps","result"],
        ["misconception_id","rationale"],
        ["rejected"],
      ]
    (root/"results.jsonl").write_text(_stable(results[0])+"\n")
    candidate_doc=json.loads(candidate.read_text())
    candidate_doc["results_sha256"]=__import__("hashlib").sha256(
      (root/"results.jsonl").read_bytes()).hexdigest()
    candidate.write_text(json.dumps(candidate_doc))
    report=verify_full_gate_d_transaction(root,expected_stages=stages,
      expected_binding_sha256=H,candidate_path=candidate)
    assert report["verified"]

@pytest.mark.parametrize("target",[
  "intent_request","policy","decision","journal","checkpoint","telemetry",
  "stage_prefix","raw_request","preflight_raw","model","token_observation",
  "gpu_observation","kv_observation","postflight","reasoning","projection",
  "backend_config",
])
def test_functional_full_reconciliation_rejects_exact_mutations(tmp_path,target):
    root,candidate,stages=_functional_transaction_fixture(tmp_path)
    if target=="intent_request":
      p=root/"call-intents.jsonl"; rows_=p.read_text().splitlines()
      row=json.loads(rows_[0]); row["request_sha256"]="b"*64; rows_[0]=_stable(row)
      p.write_text("\n".join(rows_)+"\n")
    elif target=="policy":
      p=root/"gate-d-request-schedule.json"; doc=json.loads(p.read_text())
      doc["policy"]["limits"]["semantic_repairs_per_stage"]=1
      p.write_text(json.dumps(doc))
      c=json.loads(candidate.read_text()); c["schedule_sha256"]=__import__("hashlib").sha256(p.read_bytes()).hexdigest()
      candidate.write_text(json.dumps(c))
    elif target=="decision":
      p=root/"audit/decision-capture/events.jsonl"; rows_=p.read_text().splitlines()
      row=json.loads(rows_[0]); context=json.loads(row["context"])
      context["attempt"]=2; row["context"]=_stable(context); rows_[0]=_stable(row)
      p.write_text("\n".join(rows_)+"\n")
    elif target=="journal":
      p=root/"micro-journals/unit.jsonl"; rows_=p.read_text().splitlines()
      row=json.loads(rows_[2]); row["slot"]=1; rows_[2]=_stable(row)
      p.write_text("\n".join(rows_)+"\n")
    elif target=="checkpoint":
      (root/"run.checkpoint.jsonl").write_text("")
    elif target=="telemetry":
      p=root/"telemetry/summary.json"; doc=json.loads(p.read_text())
      doc["reasoning_bytes"]=1; p.write_text(json.dumps(doc))
    elif target in {"token_observation","gpu_observation","kv_observation"}:
      p=root/"telemetry/summary.json"; doc=json.loads(p.read_text())
      doc[{"token_observation":"token_observations",
           "gpu_observation":"gpu_observations",
           "kv_observation":"kv_observations"}[target]]=[]
      p.write_text(json.dumps(doc))
    elif target=="model":
      p=root/"call-intents.jsonl"; rows_=p.read_text().splitlines()
      row=json.loads(rows_[0]); row["model"]="foreign"; rows_[0]=_stable(row)
      p.write_text("\n".join(rows_)+"\n")
    elif target=="postflight":
      p=root/"functional-postflight/postflight.json"; doc=json.loads(p.read_text())
      doc["active_clients"]=1; p.write_text(json.dumps(doc))
    elif target=="reasoning":
      p=root/"audit/http-raw/response-0.json"
      p.write_text(json.dumps({"choices":[{"message":{
        "reasoning_content":"hidden"}}]}))
    elif target=="projection":
      p=root/"results.jsonl"; doc=json.loads(p.read_text())
      doc["projection_contract"]="foreign"; p.write_text(_stable(doc)+"\n")
      c=json.loads(candidate.read_text()); c["results_sha256"]=__import__("hashlib").sha256(p.read_bytes()).hexdigest()
      candidate.write_text(json.dumps(c))
    elif target=="backend_config":
      p=root/"functional-preflight/raw/backend-config.json"
      doc=json.loads(p.read_text()); doc["model_revision"]="foreign"
      p.write_text(_stable(doc))
    elif target=="stage_prefix":
      p=root/"call-intents.jsonl"; rows_=p.read_text().splitlines()
      row=json.loads(rows_[0]); row["stage"]="foreign_A"; rows_[0]=_stable(row)
      p.write_text("\n".join(rows_)+"\n")
    elif target=="raw_request":
      p=root/"audit/http-raw/request-2.json"; p.write_bytes(b"{}")
    else:
      p=root/"functional-preflight/raw/models-second.bin"; p.write_bytes(b"{}")
    with pytest.raises(ValueError):
      verify_full_gate_d_transaction(root,expected_stages=stages,
        expected_binding_sha256=H,candidate_path=candidate)

def test_full_transaction_verifier_accepts_recomputed_artifacts(tmp_path):
    root,candidate,stages=_transaction_fixture(tmp_path)
    report=verify_full_gate_d_transaction(root,expected_stages=stages,
      expected_binding_sha256=H,candidate_path=candidate)
    assert report["verified"] and report["stage_count"]==5

@pytest.mark.parametrize("mutation",["missing","extra","reorder","candidate"])
def test_full_transaction_verifier_rejects_artifact_mutation(tmp_path,mutation):
    root,candidate,stages=_transaction_fixture(tmp_path)
    if mutation=="missing": (root/"telemetry/summary.json").unlink()
    elif mutation=="extra":
        with (root/"http_attempts.jsonl").open("a") as f:
            f.write(json.dumps({"event":"http_attempt_started","stage":"extra",
              "attempt":1})+"\n")
    elif mutation=="reorder":
        rows=(root/"call-intents.jsonl").read_text().splitlines()
        (root/"call-intents.jsonl").write_text("\n".join(reversed(rows))+"\n")
    else:
        d=json.loads(candidate.read_text()); d["results_sha256"]="0"*64
        candidate.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        verify_full_gate_d_transaction(root,expected_stages=stages,
          expected_binding_sha256=H,candidate_path=candidate)

@pytest.mark.parametrize("relative",[
 "call-intents.jsonl","http_attempts.jsonl","audit/http-raw/request-0.json",
 "audit/http-raw/response-0.json","audit/decision-capture/events.jsonl",
 "micro-journals/unit.jsonl","run.checkpoint.jsonl",
 "telemetry/summary.json","results.jsonl","summary.json",
 "gate-d-consumption.json","gate-d-request-schedule.json",
])
def test_full_transaction_verifier_rejects_each_missing_artifact(tmp_path,relative):
    root,candidate,stages=_transaction_fixture(tmp_path)
    (root/relative).unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        verify_full_gate_d_transaction(root,expected_stages=stages,
          expected_binding_sha256=H,candidate_path=candidate)
