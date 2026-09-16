#!/usr/bin/env python3
"""Upload and submit exactly the two user-authorized, verified candidates.

Run with the existing Synapse SDK environment; never logs credentials.
No emails or team messages are sent. Receipts are persisted after each write.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import synapseclient
from synapseclient.core.exceptions import SynapseHTTPError

ROOT = Path(__file__).resolve().parents[2]


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def eligible(syn, evaluation):
    info = syn.restGET(f"/evaluation/{evaluation}/team/3602194/submissionEligibility")
    if not info["teamEligibility"]["isEligible"]:
        raise RuntimeError(f"Team ineligible for queue {evaluation}: {info['teamEligibility']}")


def receipt(submission):
    return {k: submission.get(k) for k in (
        "id", "evaluationId", "entityId", "versionNumber", "name", "teamId", "createdOn")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    candidates = plan["candidates"]
    assert len(candidates) == 2 and {c["arm"] for c in candidates} == {"aug", "geo"}
    assert candidates[-1]["arm"] == plan["final_candidate"]
    for c in candidates:
        assert c["evaluation_id"] == "9619534" and c["team_id"] == "3602194"
        assert c["parent_id"] == "syn77263888" and c["audit"]["smoke_passed"]
        assert c["summary"]["folds"] == 5 and len(c["summary"]["checkpoints"]) == 5
        assert sha256(Path(c["archive"])) == c["sha256"]
    out = args.plan.parent
    journal = out / "submission_receipts.json"
    if journal.exists():
        raise FileExistsError("Submission journal already exists; inspect it before any retry")
    syn = synapseclient.Synapse()
    syn.login(silent=True)
    eligible(syn, "9619534")
    state = {"status": "running", "started_unix": time.time(), "candidates": [],
             "final_candidate": plan["final_candidate"], "notification_email_sent": False}
    save(journal, state)
    method = (ROOT / "full_version/geosurg_ic/METHOD_SUBMISSION.md").read_text()
    for candidate in candidates:
        row = {"arm": candidate["arm"], "version": candidate["version"],
               "sha256": candidate["sha256"], "archive": candidate["archive"]}
        state["candidates"].append(row)
        save(journal, state)
        entity = syn.store(synapseclient.File(
            candidate["archive"], parent="syn77263888",
            annotations={"sha256": candidate["sha256"], "task": "Task 2",
                         "method_variant": candidate["arm"], "folds": 5,
                         "training_dataset": "TIGER corrected 517-frame snapshot",
                         "pretrained_checkpoints": [
                             "facebook/mask2former-swin-small-coco-panoptic",
                             "depth-anything/Depth-Anything-V2-Small-hf (training-only geometry/diagnostics)"]}),
            createOrUpdate=False)
        row["entity_id"] = entity.id
        row["entity_version"] = entity.versionNumber
        save(journal, state)
        markdown = (f"# {candidate['version']}\n\nThis archive contains the **{candidate['arm']}** "
                    f"candidate only. Task 2, five-fold ensemble, 640 x 1120 input.\n\n"
                    f"Archive SHA-256: `{candidate['sha256']}`.\n\n" + method)
        wiki = syn.store(synapseclient.Wiki(owner=entity.id, title=candidate["version"], markdown=markdown))
        row["method_wiki_id"] = wiki.id
        save(journal, state)
        eligible(syn, "9619534")
        submission = syn.submit("9619534", entity, name=f"Genesis_CVMAIL_{candidate['version']}",
                                team="3602194", submitterAlias="Genesis_CVMAIL", silent=True)
        row["submission"] = receipt(submission)
        save(journal, state)
        try:
            status = syn.getSubmissionStatus(submission.id)
            row["evaluation_status"] = status.get("status")
        except SynapseHTTPError as error:
            # Submission receipt already exists; never retry a POST because
            # private scoring annotations are unavailable to the participant.
            row["evaluation_status"] = "status_read_unavailable"
            row["status_http_code"] = getattr(error.response, "status_code", None)
        save(journal, state)
        print(json.dumps({"submitted": row}), flush=True)
    # Publish the required technical write-up without replacing any existing wiki.
    links = "\n".join(f"- {r['version']}: https://www.synapse.org/Synapse:{r['entity_id']} "
                      f"(Task-2 submission {r['submission']['id']})" for r in state["candidates"])
    final = state["candidates"][-1]
    final_note = (f"\n\n## Submitted artifacts\n\n{links}\n\n"
                  f"Final Task-2 candidate: **{final['version']}**, submission **{final['submission']['id']}**. "
                  "Only this final Task-2 submission is intended for official scoring.\n")
    try:
        existing = syn.getWiki("syn77263888")
    except SynapseHTTPError as error:
        if getattr(error.response, "status_code", None) != 404:
            raise
        existing = None
    kwargs = {"owner": "syn77263888", "title": "Task 2: D517 decoder adaptation — final submission",
              "markdown": method + final_note}
    if existing is not None:
        kwargs["parentWikiId"] = existing.id
    wiki = syn.store(synapseclient.Wiki(**kwargs))
    state["project_method_wiki_id"] = wiki.id
    save(journal, state)
    eligible(syn, "9620081")
    final_entity = syn.get(final["entity_id"], downloadFile=False)
    writeup = syn.submit("9620081", final_entity,
        name="Genesis_CVMAIL_Task2_D517_Adaptation_WriteUp", team="3602194",
        submitterAlias="Genesis_CVMAIL", silent=True)
    state["writeup_submission"] = receipt(writeup)
    state["status"] = "completed"
    state["finished_unix"] = time.time()
    save(journal, state)
    print(json.dumps({"complete": True, "final": final["submission"],
                      "writeup": state["writeup_submission"]}), flush=True)


if __name__ == "__main__":
    main()
