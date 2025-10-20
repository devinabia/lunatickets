# services/qdrant_service.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from bs4 import BeautifulSoup
from openai import OpenAI
import asyncio
import uuid
import time
from typing import List, Dict, Any
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv


class QdrantService:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=False,
            check_compatibility=False,
        )

        # Replace SentenceTransformer with OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.collection_name = os.getenv("QDRANT_COLLECTION")

        # Use the working credentials format
        self.confluence_base_url = f"{os.getenv("JIRA_BASE_URL")}wiki"
        self.jira_base_url = os.getenv("JIRA_BASE_URL")
        self.auth = (os.getenv("JIRA_EMAIL"), os.getenv("JIRA_TOKEN"))

        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=[
                "\n\n",  # Paragraphs (highest priority)
                "\n",  # Lines
                " ",  # Words
                ".",  # Sentences
                ",",  # Clauses
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",  # Characters (lowest priority)
            ],
            keep_separator=True,  # Keep separators for better context
        )

    async def _get_embedding(self, text: str) -> list:
        """Get embedding using OpenAI API with budget-friendly settings"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                input=text,
                model="text-embedding-3-small",  # Budget-friendly choice
                dimensions=1024,  # Sweet spot for quality vs cost
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error getting embedding: {e}")
            raise

    def _get_embeddings_batch(self, texts: List[str]) -> List[list]:
        """Get embeddings for multiple texts in a single API call"""
        try:
            # OpenAI allows up to 2048 inputs per request for embedding-3-small
            batch_size = 100  # Conservative batch size to avoid rate limits
            embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                response = self.openai_client.embeddings.create(
                    input=batch, model="text-embedding-3-small", dimensions=1024
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

                # Rate limiting - be nice to the API
                if i + batch_size < len(texts):
                    time.sleep(0.1)

                logging.info(
                    f"Processed embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
                )

            return embeddings
        except Exception as e:
            logging.error(f"Error getting batch embeddings: {e}")
            raise

    def create_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1024,  # Updated to match OpenAI embedding dimensions
                        distance=Distance.COSINE,
                    ),
                )
                logging.info(f"Created collection: {self.collection_name}")
            else:
                logging.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logging.error(f"Error creating collection: {e}")
            raise

    def test_confluence_connection(self):
        """Test Confluence connection using the working method"""
        try:
            import requests

            logging.info(
                f"Testing Confluence connection to: {self.confluence_base_url}"
            )
            logging.info(f"Using auth: {self.auth[0]}")

            # Test with the same method that works
            url = f"{self.confluence_base_url}/rest/api/space"
            response = requests.get(url, auth=self.auth, timeout=30)

            if response.status_code == 200:
                data = response.json()
                spaces_count = len(data.get("results", []))
                logging.info(
                    f"✅ Confluence connection successful! Found {spaces_count} spaces"
                )
                return True
            else:
                logging.error(
                    f"❌ Confluence connection failed: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            logging.error(f"❌ Confluence connection error: {type(e).__name__}: {e}")
            return False

    def test_jira_connection(self):
        """Test Jira connection using the working method"""
        try:
            import requests

            logging.info(f"Testing Jira connection to: {self.jira_base_url}")
            logging.info(f"Using auth: {self.auth[0]}")

            # Test with direct API call
            url = f"{self.jira_base_url}/rest/api/2/myself"
            response = requests.get(url, auth=self.auth, timeout=30)

            if response.status_code == 200:
                data = response.json()
                display_name = data.get("displayName", "Unknown")
                logging.info(
                    f"✅ Jira connection successful! Connected as: {display_name}"
                )
                return True
            else:
                logging.error(
                    f"❌ Jira connection failed: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            logging.error(f"❌ Jira connection error: {type(e).__name__}: {e}")
            return False

    def extract_confluence_data(self) -> List[Dict[str, Any]]:
        """Extract data from all accessible Confluence spaces"""
        try:
            import requests

            logging.info("Starting Confluence data extraction from all spaces...")

            # Test connection first
            connection_test_passed = self.test_confluence_connection()
            if not connection_test_passed:
                logging.warning(
                    "Connection test failed, but proceeding with data extraction attempt..."
                )

            documents = []

            # Get all spaces first
            logging.info("Fetching all available spaces...")
            spaces_url = f"{self.confluence_base_url}/rest/api/space?limit=1000"
            spaces_response = requests.get(spaces_url, auth=self.auth, timeout=30)
            spaces_response.raise_for_status()

            spaces_data = spaces_response.json()
            all_spaces = spaces_data.get("results", [])
            logging.info(f"Found {len(all_spaces)} spaces in Confluence")

            for space in all_spaces:
                space_key = space["key"]
                space_name = space.get("name", space_key)

                logging.info(f"Processing space: {space_name} ({space_key})")

                try:
                    # Get all pages from current space
                    url = f"{self.confluence_base_url}/rest/api/space/{space_key}/content/page?limit=1000"
                    response = requests.get(url, auth=self.auth, timeout=30)

                    if response.status_code != 200:
                        logging.warning(
                            f"Could not access pages in space {space_name}: {response.status_code}"
                        )
                        continue

                    pages_data = response.json()
                    pages = pages_data.get("results", [])
                    logging.info(f"Found {len(pages)} pages in space {space_name}")

                    for page in pages:
                        try:
                            page_id = page["id"]

                            # Get page content with body
                            content_url = f"{self.confluence_base_url}/rest/api/content/{page_id}?expand=body.storage,version"
                            content_response = requests.get(
                                content_url, auth=self.auth, timeout=30
                            )
                            content_response.raise_for_status()

                            page_content = content_response.json()

                            if (
                                "body" in page_content
                                and "storage" in page_content["body"]
                            ):
                                html_content = page_content["body"]["storage"]["value"]
                                soup = BeautifulSoup(html_content, "html.parser")

                                # Clean HTML content
                                for script in soup(["script", "style"]):
                                    script.decompose()

                                clean_text = soup.get_text()
                                lines = (
                                    line.strip() for line in clean_text.splitlines()
                                )
                                clean_text = "\n".join(line for line in lines if line)

                                if clean_text.strip():
                                    page_url = f"{self.confluence_base_url}/spaces/{space_key}/pages/{page_id}"

                                    documents.append(
                                        {
                                            "id": str(uuid.uuid4()),
                                            "text": clean_text,
                                            "source": "confluence",
                                            "title": page_content["title"],
                                            "space": space_name,
                                            "space_key": space_key,
                                            "url": page_url,
                                            "page_id": page_id,
                                            "created_date": page_content.get(
                                                "version", {}
                                            ).get("when", ""),
                                            "type": "page",
                                        }
                                    )

                        except Exception as e:
                            logging.warning(
                                f"Error processing page {page.get('id', 'unknown')} in space {space_name}: {e}"
                            )
                            continue

                        # Rate limiting
                        time.sleep(0.1)

                except Exception as e:
                    logging.warning(f"Error processing space {space_key}: {e}")
                    continue

            logging.info(
                f"Successfully extracted {len(documents)} documents from {len(all_spaces)} Confluence spaces"
            )
            return documents

        except Exception as e:
            logging.error(f"Error in extract_confluence_data: {type(e).__name__}: {e}")
            raise

    def extract_jira_data(self) -> List[Dict[str, Any]]:
        """Extract all data from Jira using direct API calls"""
        try:
            import requests

            logging.info("Starting Jira data extraction...")

            # Test connection first
            connection_test_passed = self.test_jira_connection()
            if not connection_test_passed:
                logging.warning(
                    "Connection test failed, but proceeding with data extraction attempt..."
                )

            documents = []

            try:
                # Get all projects
                url = f"{self.jira_base_url}/rest/api/2/project"
                response = requests.get(url, auth=self.auth, timeout=30)
                response.raise_for_status()

                projects = response.json()
                logging.info(f"Found {len(projects)} Jira projects")

                for project in projects:
                    project_key = project["key"]
                    project_name = project["name"]
                    logging.info(f"Processing project: {project_name} ({project_key})")

                    try:
                        start_at = 0
                        max_results = 100
                        project_issues_count = 0

                        while True:
                            try:
                                # Search for issues in project
                                search_url = f"{self.jira_base_url}/rest/api/2/search"
                                params = {
                                    "jql": f"project = {project_key}",
                                    "startAt": start_at,
                                    "maxResults": max_results,
                                    "expand": "changelog",
                                }

                                response = requests.get(
                                    search_url,
                                    auth=self.auth,
                                    params=params,
                                    timeout=30,
                                )
                                response.raise_for_status()

                                search_results = response.json()
                                issues = search_results.get("issues", [])

                                if not issues:
                                    break

                                for issue in issues:
                                    try:
                                        fields = issue["fields"]

                                        text_content = (
                                            f"Summary: {fields.get('summary', '')}\n"
                                        )

                                        if fields.get("description"):
                                            text_content += f"Description: {fields['description']}\n"

                                        # Get comments
                                        try:
                                            comments_url = f"{self.jira_base_url}/rest/api/2/issue/{issue['key']}/comment"
                                            comments_response = requests.get(
                                                comments_url, auth=self.auth, timeout=30
                                            )

                                            if comments_response.status_code == 200:
                                                comments_data = comments_response.json()
                                                comments = comments_data.get(
                                                    "comments", []
                                                )

                                                if comments:
                                                    text_content += "Comments:\n"
                                                    for comment in comments:
                                                        text_content += f"- {comment.get('body', '')}\n"
                                        except Exception as e:
                                            logging.debug(
                                                f"Error fetching comments for {issue['key']}: {e}"
                                            )

                                        if text_content.strip():
                                            issue_url = f"{self.jira_base_url}/browse/{issue['key']}"

                                            documents.append(
                                                {
                                                    "id": str(uuid.uuid4()),
                                                    "text": text_content,
                                                    "source": "jira",
                                                    "title": fields.get("summary", ""),
                                                    "project": project_name,
                                                    "project_key": project_key,
                                                    "issue_key": issue["key"],
                                                    "issue_type": fields.get(
                                                        "issuetype", {}
                                                    ).get("name", ""),
                                                    "status": fields.get(
                                                        "status", {}
                                                    ).get("name", ""),
                                                    "priority": fields.get(
                                                        "priority", {}
                                                    ).get("name", ""),
                                                    "assignee": (
                                                        fields.get("assignee", {}).get(
                                                            "displayName", ""
                                                        )
                                                        if fields.get("assignee")
                                                        else ""
                                                    ),
                                                    "url": issue_url,
                                                    "created_date": fields.get(
                                                        "created", ""
                                                    ),
                                                    "updated_date": fields.get(
                                                        "updated", ""
                                                    ),
                                                    "type": "issue",
                                                }
                                            )
                                            project_issues_count += 1

                                    except Exception as e:
                                        logging.warning(
                                            f"Error processing issue {issue.get('key', 'unknown')}: {e}"
                                        )
                                        continue

                                start_at += max_results
                                time.sleep(0.1)  # Rate limiting

                            except Exception as e:
                                logging.warning(
                                    f"Error fetching issues for project {project_key} at offset {start_at}: {e}"
                                )
                                break

                        logging.info(
                            f"Processed {project_issues_count} issues from project {project_name}"
                        )

                    except Exception as e:
                        logging.warning(f"Error processing project {project_key}: {e}")
                        continue

            except Exception as e:
                logging.error(f"Error fetching Jira projects: {e}")
                raise

            logging.info(f"Successfully extracted {len(documents)} documents from Jira")
            return documents

        except Exception as e:
            logging.error(f"Error in extract_jira_data: {type(e).__name__}: {e}")
            raise

    def chunk_text_with_langchain(self, text: str, title: str = "") -> List[str]:
        """
        Single chunking function using LangChain RecursiveCharacterTextSplitter

        Args:
            text: Text to chunk
            title: Optional document title for context

        Returns:
            List of chunked text with optional context
        """
        try:
            # Clean the text
            text = text.strip()
            if not text:
                return []

            # For very short documents, don't split at all - keep them whole
            if len(text) <= 200:
                if title and title.strip():
                    return [f"Document: {title}\n\n{text}"]
                return [text]

            # Use LangChain text splitter for longer documents
            chunks = self.text_splitter.split_text(text)

            # Don't filter out ANY chunks - keep everything, even short ones
            filtered_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

            # Add document context to each chunk for better searchability
            if title and title.strip():
                contextual_chunks = []
                for chunk in filtered_chunks:
                    # Add title context at the beginning of each chunk
                    contextual_chunk = f"Document: {title}\n\n{chunk}"
                    contextual_chunks.append(contextual_chunk)
                return contextual_chunks

            return filtered_chunks

        except Exception as e:
            logging.error(f"Error in LangChain chunking: {e}")
            # Fallback to keep the original text intact
            if title and title.strip():
                return [f"Document: {title}\n\n{text}"]
            return [text]

    def clear_collection(self):
        """Delete and recreate the collection"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name in collection_names:
                self.qdrant_client.delete_collection(self.collection_name)
                logging.info(f"Deleted existing collection: {self.collection_name}")

        except Exception as e:
            logging.warning(f"Error clearing collection: {e}")
            raise

    def dump_all_data_to_qdrant(self) -> Dict[str, Any]:
        """Dump both Confluence AND Jira data to Qdrant (deletes and recreates collection)"""
        try:
            logging.info("Starting data dump to Qdrant with LangChain chunking...")

            # Extract data from BOTH sources
            confluence_docs = []
            jira_docs = []

            try:
                logging.info("Extracting Confluence data...")
                confluence_docs = self.extract_confluence_data()
            except Exception as e:
                logging.error(f"Failed to extract Confluence data: {e}")
                # Don't raise - continue with Jira even if Confluence fails

            try:
                logging.info("Extracting Jira data...")
                jira_docs = self.extract_jira_data()
            except Exception as e:
                logging.error(f"Failed to extract Jira data: {e}")
                # Don't raise - continue with Confluence even if Jira fails

            all_documents = confluence_docs + jira_docs

            if not all_documents:
                return {
                    "status": "warning",
                    "message": "No documents found to index",
                    "total_documents": 0,
                    "confluence_documents": 0,
                    "jira_documents": 0,
                }

            # Process documents into chunks and create embeddings
            logging.info(
                f"Processing {len(all_documents)} documents into chunks with LangChain..."
            )
            logging.info(
                f"Breakdown: {len(confluence_docs)} Confluence + {len(jira_docs)} Jira documents"
            )
            points = []
            all_chunks = []
            chunk_metadata = []

            # First pass: collect all chunks using LangChain
            for doc_idx, doc in enumerate(all_documents):
                try:
                    # Use LangChain chunking with document context
                    chunks = self.chunk_text_with_langchain(
                        doc["text"], title=doc.get("title", "")
                    )

                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            all_chunks.append(chunk)
                            chunk_metadata.append(
                                {
                                    **doc,
                                    "chunk_index": i,
                                    "total_chunks": len(chunks),
                                }
                            )

                    if (doc_idx + 1) % 10 == 0:
                        logging.info(
                            f"Processed {doc_idx + 1}/{len(all_documents)} documents"
                        )

                except Exception as e:
                    logging.warning(
                        f"Error processing document {doc.get('title', 'unknown')}: {e}"
                    )
                    continue

            # Second pass: get embeddings in batches (more efficient)
            logging.info(f"Getting embeddings for {len(all_chunks)} chunks...")
            embeddings = self._get_embeddings_batch(all_chunks)

            # Third pass: create points
            for i, (chunk, metadata, embedding) in enumerate(
                zip(all_chunks, chunk_metadata, embeddings)
            ):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        **metadata,
                        "text": chunk,
                    },
                )
                points.append(point)

            self.clear_collection()
            self.create_collection()

            # Upload to Qdrant in batches
            logging.info(f"Uploading {len(points)} chunks to Qdrant...")
            batch_size = 100
            total_points = len(points)

            for i in range(0, total_points, batch_size):
                try:
                    batch = points[i : i + batch_size]
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name, points=batch
                    )
                    logging.info(
                        f"Uploaded batch {i//batch_size + 1}/{(total_points + batch_size - 1)//batch_size}"
                    )
                except Exception as e:
                    logging.error(f"Error uploading batch {i//batch_size + 1}: {e}")
                    raise

            logging.info("Data dump completed successfully with LangChain chunking!")
            return {
                "status": "success",
                "message": "Both Confluence and Jira data successfully dumped to Qdrant with LangChain chunking",
                "total_documents": len(all_documents),
                "total_chunks": total_points,
                "confluence_documents": len(confluence_docs),
                "jira_documents": len(jira_docs),
                "collection_name": self.collection_name,
            }

        except Exception as e:
            logging.error(f"Error dumping data to Qdrant: {type(e).__name__}: {e}")
            return {
                "status": "error",
                "message": f"Failed to dump data: {str(e)}",
                "total_documents": 0,
                "total_chunks": 0,
                "confluence_documents": 0,
                "jira_documents": 0,
            }
