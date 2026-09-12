import gradio as gr

from agent import run_hybrid_agent
from ingest import index_documents
from tools_rag import has_documents


# =========================================================
# PDF UPLOAD / INDEXING
# =========================================================

def upload_pdf(pdf_file):

    if pdf_file is None:
        return "❌ Please select a PDF."

    try:
        chunks = index_documents([pdf_file])

        return (
            f"✅ PDF indexed successfully — "
            f"{chunks} chunks."
        )

    except Exception as e:

        return f"❌ PDF indexing error: {e}"


# =========================================================
# CHAT
# =========================================================

def chat(message, image_file, history):

    if not message:
        return history, ""

    try:

        # -------------------------------------------------
        # Check whether a PDF is currently indexed
        # -------------------------------------------------

        pdf_available = has_documents()

        # -------------------------------------------------
        # Run HybridSight
        #
        # The PLANNER LLM decides which tools are needed.
        # No keyword-based routing is used here.
        # -------------------------------------------------

        response = run_hybrid_agent(
            question=message,
            pdf_available=pdf_available,
            image_path=image_file,
        )

        # -------------------------------------------------
        # Add user message
        # -------------------------------------------------

        history.append(
            {
                "role": "user",
                "content": message,
            }
        )

        # -------------------------------------------------
        # Add assistant response
        # -------------------------------------------------

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
# GRADIO INTERFACE
# =========================================================

with gr.Blocks(
    title="HybridSight"
) as demo:

    # =====================================================
    # HEADER
    # =====================================================

    gr.Markdown(
        """
        # 🔭 HybridSight

        ### AI That Reads, Searches, and Sees

        HybridSight is a hybrid AI agent capable of
        combining multiple information sources.

        **Capabilities**

        📄 **PDF / RAG** — Search uploaded documents  
        🌐 **Web Search** — Search current information  
        📚 **Wikipedia** — General knowledge  
        🖼️ **Vision** — Analyze uploaded images  
        🧠 **Hybrid Reasoning** — Combine multiple tools
        """
    )

    # =====================================================
    # MAIN LAYOUT
    # =====================================================

    with gr.Row():

        # =================================================
        # LEFT PANEL
        # =================================================

        with gr.Column(scale=1):

            # ---------------------------------------------
            # PDF SECTION
            # ---------------------------------------------

            gr.Markdown("## 📄 PDF / RAG")

            pdf_file = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath",
            )

            upload_button = gr.Button(
                "Index PDF"
            )

            upload_status = gr.Textbox(
                label="PDF Status",
                interactive=False,
            )

            upload_button.click(
                upload_pdf,
                inputs=pdf_file,
                outputs=upload_status,
            )

            # ---------------------------------------------
            # IMAGE SECTION
            # ---------------------------------------------

            gr.Markdown("## 🖼️ Vision")

            image_file = gr.Image(
                label="Upload Image",
                type="filepath",
            )

            gr.Markdown(
                """
                **Tip:** You can upload both a PDF
                and an image.

                HybridSight can use information from
                **both sources** when the question
                requires it.
                """
            )

        # =================================================
        # RIGHT PANEL
        # =================================================

        with gr.Column(scale=2):

            # ---------------------------------------------
            # CHATBOT
            # ---------------------------------------------

            chatbot = gr.Chatbot(
                label="HybridSight",
                height=500,
            )

            # ---------------------------------------------
            # MESSAGE BOX
            # ---------------------------------------------

            message = gr.Textbox(
                label="Ask HybridSight",
                placeholder=(
                    "Ask about your PDF, current events, "
                    "general knowledge, an image, or "
                    "combine PDF + image..."
                ),
            )

            # ---------------------------------------------
            # SEND BUTTON
            # ---------------------------------------------

            send_button = gr.Button(
                "Send",
                variant="primary",
            )

            # ---------------------------------------------
            # SEND BUTTON EVENT
            # ---------------------------------------------

            send_button.click(
                chat,
                inputs=[
                    message,
                    image_file,
                    chatbot,
                ],
                outputs=[
                    chatbot,
                    message,
                ],
            )

            # ---------------------------------------------
            # ENTER KEY EVENT
            # ---------------------------------------------

            message.submit(
                chat,
                inputs=[
                    message,
                    image_file,
                    chatbot,
                ],
                outputs=[
                    chatbot,
                    message,
                ],
            )


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    demo.launch()