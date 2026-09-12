import gradio as gr

from agent import run_hybrid_agent
from ingest import index_documents
from tools_rag import clear_documents


# =========================================================
# PDF INDEXING
# =========================================================

def upload_pdf(
    pdf_file,
    progress=gr.Progress(),
):

    if pdf_file is None:

        progress(
            1.0,
            desc="❌ Please select a PDF."
        )

        return (
            "❌ Please select a PDF.",
            False,
        )


    try:

        progress(
            0.0,
            desc="🚀 Starting PDF indexing..."
        )


        chunks = index_documents(
            [pdf_file],
            progress=progress,
        )


        if chunks == 0:

            return (
                "❌ No content could be extracted from the PDF.",
                False,
            )


        return (
            f"✅ PDF indexed successfully — "
            f"{chunks} chunks.",
            True,
        )


    except Exception as e:

        progress(
            1.0,
            desc="❌ PDF indexing failed."
        )

        return (
            f"❌ PDF indexing error: {e}",
            False,
        )
# =========================================================
# PDF SELECTED
# =========================================================

def pdf_selected(pdf_file):

    if pdf_file is None:
        return "🗑️ No PDF selected.", False

    return (
        "📄 PDF selected. Click 'Index PDF' to make it available.",
        False,
    )


# =========================================================
# PDF REMOVED
# =========================================================

def clear_pdf():

    try:
        clear_documents()

        return (
            "🗑️ PDF removed. Document index cleared.",
            False,
        )

    except Exception as e:
        return (
            f"❌ PDF removal error: {e}",
            False,
        )


# =========================================================
# MAIN HYBRID CHAT
# =========================================================

def chat(
    message,
    image_file,
    pdf_indexed,
    history,
):

    if not message or not message.strip():
        return history, ""

    try:

        response = run_hybrid_agent(
            question=message,
            pdf_available=bool(pdf_indexed),
            image_path=image_file,
        )

        history.append(
            {
                "role": "user",
                "content": message,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        return history, ""

    except Exception as e:

        history.append(
            {
                "role": "user",
                "content": message,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": f"❌ Agent error: {e}",
            }
        )

        return history, ""


# =========================================================
# DOCUMENT QA
# =========================================================

def document_chat(
    message,
    history,
    pdf_indexed,
):

    if not message or not message.strip():
        return history, ""

    try:

        if not pdf_indexed:

            answer = (
                "📄 No PDF is currently indexed. "
                "Please upload and index a PDF first."
            )

        else:

            answer = run_hybrid_agent(
                question=message,
                pdf_available=True,
                image_path=None,
            )

        history.append(
            {
                "role": "user",
                "content": message,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return history, ""

    except Exception as e:

        history.append(
            {
                "role": "user",
                "content": message,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": f"❌ Document error: {e}",
            }
        )

        return history, ""


# =========================================================
# IMAGE ANALYSIS
# =========================================================

def analyze_image(
    image_file,
):

    if image_file is None:
        return "🖼️ Please upload an image first."

    try:

        response = run_hybrid_agent(
            question="Describe this image and identify its important details.",
            pdf_available=False,
            image_path=image_file,
        )

        return response

    except Exception as e:

        return f"❌ Vision error: {e}"


# =========================================================
# UI
# =========================================================

with gr.Blocks(
    title="HybridSight — GenAI Portfolio App",
    theme=gr.themes.Soft(),
) as demo:

    # =====================================================
    # SESSION STATE
    # =====================================================

    pdf_indexed = gr.State(False)


    # =====================================================
    # HEADER
    # =====================================================

    gr.Markdown(
        """
        # 🔭 HybridSight

        ### AI That Reads, Searches, and Sees

        **RAG + Web Search + Wikipedia + Vision**

        A hybrid GenAI agent that intelligently selects
        the right information source for your question.
        """
    )


    # =====================================================
    # TABS
    # =====================================================

    with gr.Tabs():

        # =================================================
        # TAB 1 — HYBRID CHAT
        # =================================================

        with gr.Tab("💬 Hybrid Chat"):

            with gr.Row():

                # -----------------------------------------
                # CHAT
                # -----------------------------------------

                with gr.Column(
                    scale=3
                ):

                    chatbot = gr.Chatbot(
                        height=440,
                        label="Conversation",
                    )

                    msg_box = gr.Textbox(
                        placeholder="Ask anything...",
                        show_label=False,
                    )

                    send_button = gr.Button(
                        "Send",
                        variant="primary",
                    )


                # -----------------------------------------
                # SIDEBAR
                # -----------------------------------------

                with gr.Column(
                    scale=1
                ):

                    gr.Markdown(
                        "### 🔍 Agent Information"
                    )

                    gr.Markdown(
                        """
                        HybridSight can use:

                        📄 **PDF / RAG**

                        🌐 **Web Search**

                        📚 **Wikipedia**

                        🖼️ **Vision**

                        Upload a PDF or image in the
                        appropriate tab before asking
                        questions that require it.
                        """
                    )


            # ---------------------------------------------
            # IMAGE FOR HYBRID CHAT
            # ---------------------------------------------

            gr.Markdown(
                "### 🖼️ Optional Image"
            )

            chat_image = gr.Image(
                label="Upload image for this conversation",
                type="filepath",
            )


            # ---------------------------------------------
            # SEND
            # ---------------------------------------------

            send_button.click(
                chat,
                inputs=[
                    msg_box,
                    chat_image,
                    pdf_indexed,
                    chatbot,
                ],
                outputs=[
                    chatbot,
                    msg_box,
                ],
            )


            msg_box.submit(
                chat,
                inputs=[
                    msg_box,
                    chat_image,
                    pdf_indexed,
                    chatbot,
                ],
                outputs=[
                    chatbot,
                    msg_box,
                ],
            )


        # =================================================
        # TAB 2 — DOCUMENT QA
        # =================================================

        with gr.Tab("📄 Document QA"):

            gr.Markdown(
                """
                ## 📄 Document Question Answering

                Upload a PDF, index it, then ask questions
                specifically about its contents.
                """
            )


            with gr.Row():

                with gr.Column(
                    scale=2
                ):

                    pdf_upload = gr.File(
                        label="Upload a PDF",
                        file_types=[".pdf"],
                        type="filepath",
                    )


                with gr.Column(
                    scale=1
                ):

                    index_button = gr.Button(
                        "Index PDF",
                        variant="primary",
                    )

                    index_status = gr.Textbox(
                        label="Indexing Status",
                        interactive=False,
                    )


            # ---------------------------------------------
            # PDF SELECTED
            # ---------------------------------------------

            pdf_upload.change(
                pdf_selected,
                inputs=pdf_upload,
                outputs=[
                    index_status,
                    pdf_indexed,
                ],
            )


            # ---------------------------------------------
            # INDEX PDF
            # ---------------------------------------------

            index_button.click(
                upload_pdf,
                inputs=pdf_upload,
                outputs=[
                    index_status,
                    pdf_indexed,
                ],
            )


            # ---------------------------------------------
            # REMOVE PDF
            # ---------------------------------------------

            pdf_upload.clear(
                clear_pdf,
                inputs=None,
                outputs=[
                    index_status,
                    pdf_indexed,
                ],
            )


            # ---------------------------------------------
            # DOCUMENT CHAT
            # ---------------------------------------------

            gr.Markdown(
                "### Ask About Your Document"
            )


            doc_chatbot = gr.Chatbot(
                height=380,
                label="Document Conversation",
            )


            doc_input = gr.Textbox(
                placeholder="Ask about the document...",
                show_label=False,
            )


            doc_ask_button = gr.Button(
                "Ask",
                variant="primary",
            )


            doc_ask_button.click(
                document_chat,
                inputs=[
                    doc_input,
                    doc_chatbot,
                    pdf_indexed,
                ],
                outputs=[
                    doc_chatbot,
                    doc_input,
                ],
            )


            doc_input.submit(
                document_chat,
                inputs=[
                    doc_input,
                    doc_chatbot,
                    pdf_indexed,
                ],
                outputs=[
                    doc_chatbot,
                    doc_input,
                ],
            )


        # =================================================
        # TAB 3 — IMAGE STUDIO
        # =================================================

        with gr.Tab("🖼️ Image Studio"):

            gr.Markdown(
                """
                ## 🖼️ Image Studio

                Upload an image and let HybridSight's
                vision capability analyze it.
                """
            )


            with gr.Row():

                # -----------------------------------------
                # IMAGE INPUT
                # -----------------------------------------

                with gr.Column(
                    scale=1
                ):

                    image_upload = gr.Image(
                        label="Upload Image",
                        type="filepath",
                    )


                # -----------------------------------------
                # IMAGE OUTPUT
                # -----------------------------------------

                with gr.Column(
                    scale=2
                ):

                    image_output = gr.Textbox(
                        label="Vision Analysis",
                        lines=12,
                        interactive=False,
                    )


            # ---------------------------------------------
            # ANALYZE BUTTON
            # ---------------------------------------------

            analyze_button = gr.Button(
                "🔍 Analyse Image",
                variant="primary",
            )


            analyze_button.click(
                analyze_image,
                inputs=image_upload,
                outputs=image_output,
            )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    demo.launch()