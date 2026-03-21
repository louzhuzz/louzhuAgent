from pathlib import Path

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "知识点"


def load_markdown_documents(directory: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(path.relative_to(PROJECT_ROOT)),
                    "file_name": path.name,
                    "suffix": path.suffix,
                },
            )
        )
    return documents


def main() -> None:
    documents = load_markdown_documents(KNOWLEDGE_DIR)
    print(f"已加载文档数: {len(documents)}")

    for index, document in enumerate(documents[:3], start=1):
        preview = document.page_content[:120].replace("\n", " ")
        print(f"\n文档 {index}")
        print(f"metadata: {document.metadata}")
        print(f"preview: {preview}...")


if __name__ == "__main__":
    main()

