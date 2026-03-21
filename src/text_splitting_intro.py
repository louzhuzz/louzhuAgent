from langchain_core.documents import Document

from document_loading_intro import KNOWLEDGE_DIR, load_markdown_documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0。")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0。")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size。")

    chunks: list[Document] = []
    step = chunk_size - chunk_overlap

    for document in documents:
        text = document.page_content
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = {
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "chunk_start": start,
                    "chunk_end": min(end, len(text)),
                }
                chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))

            start += step
            chunk_index += 1

    return chunks


def main() -> None:
    documents = load_markdown_documents(KNOWLEDGE_DIR)
    chunks = split_documents(documents, chunk_size=300, chunk_overlap=50)

    print(f"原始文档数: {len(documents)}")
    print(f"切分后 chunk 数: {len(chunks)}")

    for index, chunk in enumerate(chunks[:5], start=1):
        preview = chunk.page_content[:100].replace("\n", " ")
        print(f"\nChunk {index}")
        print(f"metadata: {chunk.metadata}")
        print(f"preview: {preview}...")


if __name__ == "__main__":
    main()

