"""Streamlit demo UI for the Legal AI Assistant."""

from __future__ import annotations

import streamlit as st

from legal_ai.ui.api_client import LegalApiClient


def main() -> None:
    st.set_page_config(page_title="Legal AI Assistant", page_icon=":scales:", layout="wide")
    _ensure_state()
    client = LegalApiClient()
    _render_sidebar(client)

    tab_qa, tab_risk, tab_compare = st.tabs(["Q&A", "Risk Detector", "Compare"])
    with tab_qa:
        _render_qa_tab(client)
    with tab_risk:
        _render_risk_tab(client)
    with tab_compare:
        _render_compare_tab(client)


def _ensure_state() -> None:
    st.session_state.setdefault("ingested_documents", [])
    st.session_state.setdefault("last_health", None)


def _render_sidebar(client: LegalApiClient) -> None:
    st.sidebar.title("Legal AI Assistant")
    if st.sidebar.button("Check backend health"):
        try:
            st.session_state["last_health"] = client.health()
        except Exception as exc:
            st.session_state["last_health"] = {"status": "error", "detail": str(exc)}
    health = st.session_state.get("last_health")
    if health:
        st.sidebar.json(health)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Ingest contract")
    upload = st.sidebar.file_uploader("Upload PDF", type=["pdf"], key="ingest_uploader")
    if st.sidebar.button("Index document", disabled=upload is None):
        if upload is None:
            return
        with st.sidebar, st.spinner("Indexing..."):
            try:
                result = client.ingest(upload.getvalue(), upload.name)
            except Exception as exc:
                st.error(f"Ingest failed: {exc}")
                return
        st.session_state["ingested_documents"].append(result)
        st.sidebar.success(
            f"Indexed {result['title']} ({result['chunk_count']} chunks)"
        )

    if st.session_state["ingested_documents"]:
        st.sidebar.markdown("**Indexed documents**")
        for doc in st.session_state["ingested_documents"]:
            st.sidebar.markdown(
                f"- `{doc['document_id']}` - {doc['title']} ({doc['page_count']}p / {doc['chunk_count']}c)"
            )


def _render_qa_tab(client: LegalApiClient) -> None:
    st.subheader("Question & Answer with source attribution")
    docs = st.session_state["ingested_documents"]
    options = {f"{d['title']} ({d['document_id']})": d["document_id"] for d in docs}
    selected_labels = st.multiselect("Filter by document (optional)", list(options.keys()))
    selected_ids = [options[label] for label in selected_labels]
    question = st.text_area("Question", height=100, placeholder="What is the termination notice?")
    top_k = st.slider("Top K", min_value=3, max_value=20, value=8)
    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Generating answer..."):
            try:
                response = client.qa(question, selected_ids or None, top_k)
            except Exception as exc:
                st.error(f"QA failed: {exc}")
                return
        st.markdown("### Answer")
        st.markdown(response["answer"])
        st.markdown("### Citations")
        for citation in response["citations"]:
            with st.expander(
                f"{citation['document_title']} - p.{citation['page_start']} "
                f"(score={citation['score']:.3f})"
            ):
                st.caption(f"Section: {citation['section_path'] or 'N/A'}")
                st.code(citation["snippet"])


def _render_risk_tab(client: LegalApiClient) -> None:
    st.subheader("Risk clause detector")
    docs = st.session_state["ingested_documents"]
    if not docs:
        st.info("Ingest a contract from the sidebar first.")
        return
    options = {f"{d['title']} ({d['document_id']})": d["document_id"] for d in docs}
    label = st.selectbox("Document", list(options.keys()))
    max_chunks = st.slider("Max chunks to scan", 20, 500, 200, step=20)
    if st.button("Detect risks", type="primary"):
        with st.spinner("Scanning for risks..."):
            try:
                report = client.risk(options[label], max_chunks=max_chunks)
            except Exception as exc:
                st.error(f"Risk detection failed: {exc}")
                return
        findings = report.get("findings", [])
        if not findings:
            st.success("No risks detected with the current rules.")
            return
        for finding in findings:
            color = {"high": "red", "medium": "orange", "low": "blue"}.get(finding["severity"], "gray")
            st.markdown(
                f"**[{finding['severity'].upper()}] {finding['category']}** - p.{finding['page']}"
            )
            st.caption(f"Section: {finding['section'] or 'N/A'}")
            st.markdown(f":{color}[{finding['rationale']}]")
            with st.expander("Source text and recommendation"):
                st.code(finding["source_text"])
                if finding["recommendation"]:
                    st.markdown(f"**Recommendation:** {finding['recommendation']}")


def _render_compare_tab(client: LegalApiClient) -> None:
    st.subheader("Document comparator")
    col_left, col_right = st.columns(2)
    with col_left:
        left = st.file_uploader("Left version", type=["pdf"], key="cmp_left")
    with col_right:
        right = st.file_uploader("Right version", type=["pdf"], key="cmp_right")
    if st.button("Compare", type="primary", disabled=left is None or right is None):
        if left is None or right is None:
            return
        with st.spinner("Comparing..."):
            try:
                report = client.compare(
                    left.getvalue(), left.name, right.getvalue(), right.name
                )
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")
                return
        st.metric("Total risk delta", report["total_risk_delta"])
        st.caption(
            f"{report['left_title']} ({report['total_left']} clauses) vs "
            f"{report['right_title']} ({report['total_right']} clauses)"
        )
        for diff in report["diffs"]:
            with st.expander(
                f"{diff['change_type'].upper()} - delta {diff['risk_delta']:+d} - {diff['summary'][:80]}"
            ):
                st.markdown(f"**Summary:** {diff['summary']}")
                st.markdown(f"**Rationale:** {diff['rationale']}")
                cols = st.columns(2)
                with cols[0]:
                    st.caption(f"Left p.{diff['left_page']} / {diff['left_section'] or 'N/A'}")
                    st.code(diff["left_text"] or "<missing>")
                with cols[1]:
                    st.caption(f"Right p.{diff['right_page']} / {diff['right_section'] or 'N/A'}")
                    st.code(diff["right_text"] or "<missing>")


if __name__ == "__main__":
    main()
