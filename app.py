import streamlit as st
import os
import tempfile
import requests
import zipfile
import io
import shutil
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback

# Core libraries
try:
    from PyPDF2 import PdfReader
except ImportError:
    try:
        from pypdf import PdfReader
    except ImportError:
        PdfReader = None

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader
)
from langchain.schema import Document
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from openai import OpenAI
import glob

# Page configuration
st.set_page_config(
    page_title="Kaizen Engineers AI Agent Knowledge Base",
    page_icon="🤖",
    layout="wide"
)

# GitHub ZIP URL (configure this with your repository)
GITHUB_ZIP_URL = "https://github.com/pravinraut05/Dataset/raw/main/orders.zip"

# Initialize session state variables
def initialize_session_state():
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = False
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None
    if "initialization_started" not in st.session_state:
        st.session_state.initialization_started = False
    if "documents_metadata" not in st.session_state:
        st.session_state.documents_metadata = []
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = {}
    if "tabular_data_summary" not in st.session_state:
        st.session_state.tabular_data_summary = {}

initialize_session_state()

class EnhancedDocumentProcessor:
    """Enhanced document processor with specialized tabular data handling"""
    
    @staticmethod
    def download_and_extract_zip(github_zip_url: str) -> str:
        """Download and extract ZIP file from GitHub"""
        try:
            temp_dir = tempfile.mkdtemp()
            response = requests.get(github_zip_url, timeout=30)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(temp_dir)
            
            return temp_dir
        except Exception as e:
            st.error(f"Error downloading files: {str(e)}")
            return None

    @staticmethod
    def analyze_dataframe(df: pd.DataFrame, file_name: str) -> Dict[str, Any]:
        """Comprehensive analysis of a dataframe"""
        try:
            analysis = {
                "file_name": file_name,
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "summary_stats": {},
                "sample_data": df.head(3).to_dict('records'),
                "column_descriptions": {}
            }
            
            # Generate summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                analysis["summary_stats"] = df[numeric_cols].describe().to_dict()
            
            # Generate column descriptions
            for col in df.columns:
                col_info = {
                    "type": str(df[col].dtype),
                    "unique_values": int(df[col].nunique()),
                    "null_count": int(df[col].isnull().sum())
                }
                
                if df[col].dtype == 'object':
                    # For text columns, get unique values (limited)
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) <= 10:
                        col_info["unique_values_list"] = list(unique_vals)
                    else:
                        col_info["sample_values"] = list(unique_vals[:5])
                
                analysis["column_descriptions"][col] = col_info
            
            return analysis
        
        except Exception as e:
            return {"error": str(e), "file_name": file_name}

    @staticmethod
    def create_comprehensive_text_representation(df: pd.DataFrame, file_name: str, analysis: Dict) -> str:
        """Create a comprehensive text representation of tabular data for LLM processing"""
        try:
            text_parts = []
            
            # Header information
            text_parts.append(f"=== TABULAR DATA FILE: {file_name} ===")
            text_parts.append(f"Dataset Shape: {analysis['shape'][0]} rows × {analysis['shape'][1]} columns")
            text_parts.append("")
            
            # Column information
            text_parts.append("COLUMN INFORMATION:")
            for col, info in analysis["column_descriptions"].items():
                desc = f"- {col} ({info['type']}): {info['unique_values']} unique values, {info['null_count']} missing"
                if "unique_values_list" in info:
                    desc += f", Values: {info['unique_values_list']}"
                elif "sample_values" in info:
                    desc += f", Sample values: {info['sample_values']}"
                text_parts.append(desc)
            
            text_parts.append("")
            
            # Summary statistics for numeric columns
            if analysis["summary_stats"]:
                text_parts.append("NUMERIC COLUMNS STATISTICS:")
                for col, stats in analysis["summary_stats"].items():
                    text_parts.append(f"- {col}: mean={stats.get('mean', 'N/A'):.2f}, std={stats.get('std', 'N/A'):.2f}, min={stats.get('min', 'N/A')}, max={stats.get('max', 'N/A')}")
                text_parts.append("")
            
            # Sample data
            text_parts.append("SAMPLE DATA (First 5 rows):")
            sample_df = df.head(5)
            text_parts.append(sample_df.to_string(index=False))
            text_parts.append("")
            
            # Data patterns and insights
            text_parts.append("DATA INSIGHTS:")
            
            # Missing data patterns
            missing_cols = [col for col, count in analysis["missing_values"].items() if count > 0]
            if missing_cols:
                text_parts.append(f"- Columns with missing data: {missing_cols}")
            
            # Categorical columns analysis
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                text_parts.append(f"- Categorical columns: {list(categorical_cols)}")
                for col in categorical_cols:
                    value_counts = df[col].value_counts().head(3)
                    text_parts.append(f"  * {col} top values: {dict(value_counts)}")
            
            # Numeric columns analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                text_parts.append(f"- Numeric columns: {list(numeric_cols)}")
            
            # Additional data representation for better search
            text_parts.append("")
            text_parts.append("SEARCHABLE CONTENT:")
            
            # Add all unique values for better searchability
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].nunique() < 50:
                    unique_vals = df[col].dropna().unique()
                    text_parts.append(f"{col} contains: {', '.join(map(str, unique_vals))}")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            return f"Error creating text representation for {file_name}: {str(e)}"

    @staticmethod
    def load_documents(directory_path: str) -> tuple[List[Document], List[Dict], Dict[str, pd.DataFrame], Dict[str, Dict]]:
        """Load documents with enhanced tabular data processing"""
        documents = []
        metadata_list = []
        dataframes = {}
        tabular_summaries = {}
        
        # Handle different file types
        file_handlers = {
            "**/*.pdf": EnhancedDocumentProcessor._handle_pdf,
            "**/*.txt": EnhancedDocumentProcessor._handle_text,
            "**/*.md": EnhancedDocumentProcessor._handle_markdown,
            "**/*.docx": EnhancedDocumentProcessor._handle_docx,
            "**/*.csv": EnhancedDocumentProcessor._handle_csv_enhanced,
            "**/*.xlsx": EnhancedDocumentProcessor._handle_excel_enhanced,
            "**/*.xls": EnhancedDocumentProcessor._handle_excel_enhanced,
        }
        
        for pattern, handler in file_handlers.items():
            for file_path in glob.glob(f"{directory_path}/{pattern}", recursive=True):
                try:
                    result = handler(file_path)
                    if len(result) == 4:  # Enhanced handlers return 4 values
                        docs, meta, df, summary = result
                        if df is not None:
                            dataframes[file_path] = df
                            tabular_summaries[file_path] = summary
                    else:  # Regular handlers return 2 values
                        docs, meta = result
                    
                    if docs:
                        documents.extend(docs)
                        metadata_list.extend(meta)
                        
                except Exception as e:
                    # Create a placeholder document for failed files
                    error_doc = Document(
                        page_content=f"File {os.path.basename(file_path)} could not be processed: {str(e)}",
                        metadata={"source": file_path, "file_type": "Error", "error": str(e)}
                    )
                    documents.append(error_doc)
                    metadata_list.append({"source": file_path, "file_type": "Error", "error": str(e)})
                    continue
        
        return documents, metadata_list, dataframes, tabular_summaries

    @staticmethod
    def _handle_csv_enhanced(csv_path: str) -> tuple[List[Document], List[Dict], Optional[pd.DataFrame], Optional[Dict]]:
        """Enhanced CSV handling with comprehensive analysis"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise Exception("Could not read CSV with any encoding")
            
            file_name = os.path.basename(csv_path)
            
            # Analyze the dataframe
            analysis = EnhancedDocumentProcessor.analyze_dataframe(df, file_name)
            
            # Create comprehensive text representation
            text_content = EnhancedDocumentProcessor.create_comprehensive_text_representation(df, file_name, analysis)
            
            # Create document with enhanced content
            doc = Document(
                page_content=text_content,
                metadata={
                    "source": csv_path,
                    "file_type": "CSV",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns)
                }
            )
            
            metadata = {
                "file_type": "CSV",
                "source": csv_path,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns)
            }
            
            return [doc], [metadata], df, analysis
            
        except Exception as e:
            # Fallback document
            error_doc = Document(
                page_content=f"CSV file {os.path.basename(csv_path)} could not be processed: {str(e)}",
                metadata={"source": csv_path, "file_type": "CSV", "error": str(e)}
            )
            metadata = {"source": csv_path, "file_type": "CSV", "error": str(e)}
            return [error_doc], [metadata], None, None

    @staticmethod
    def _handle_excel_enhanced(excel_path: str) -> tuple[List[Document], List[Dict], Optional[pd.DataFrame], Optional[Dict]]:
        """Enhanced Excel handling with comprehensive analysis"""
        try:
            xl = pd.ExcelFile(excel_path)
            all_docs = []
            all_metadata = []
            combined_df = pd.DataFrame()
            combined_analysis = {"sheets": {}}
            
            file_name = os.path.basename(excel_path)
            
            for sheet_name in xl.sheet_names:
                try:
                    df = pd.read_excel(excel_path, sheet_name=sheet_name)
                    
                    # Analyze this sheet
                    sheet_analysis = EnhancedDocumentProcessor.analyze_dataframe(df, f"{file_name}_{sheet_name}")
                    combined_analysis["sheets"][sheet_name] = sheet_analysis
                    
                    # Create text representation
                    text_content = EnhancedDocumentProcessor.create_comprehensive_text_representation(
                        df, f"{file_name} - Sheet: {sheet_name}", sheet_analysis
                    )
                    
                    # Create document for this sheet
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "source": excel_path,
                            "sheet_name": sheet_name,
                            "file_type": "Excel",
                            "rows": len(df),
                            "columns": len(df.columns),
                            "column_names": list(df.columns)
                        }
                    )
                    
                    all_docs.append(doc)
                    all_metadata.append({
                        "source": excel_path,
                        "sheet_name": sheet_name,
                        "file_type": "Excel",
                        "rows": len(df),
                        "columns": len(df.columns)
                    })
                    
                    # Combine data for overall analysis
                    if combined_df.empty:
                        combined_df = df.copy()
                    else:
                        # Try to concatenate if columns match
                        try:
                            combined_df = pd.concat([combined_df, df], ignore_index=True)
                        except:
                            pass  # Different schemas, keep first sheet only
                
                except Exception as e:
                    st.warning(f"Could not process sheet {sheet_name}: {str(e)}")
                    continue
            
            return all_docs, all_metadata, combined_df if not combined_df.empty else None, combined_analysis
            
        except Exception as e:
            error_doc = Document(
                page_content=f"Excel file {os.path.basename(excel_path)} could not be processed: {str(e)}",
                metadata={"source": excel_path, "file_type": "Excel", "error": str(e)}
            )
            metadata = {"source": excel_path, "file_type": "Excel", "error": str(e)}
            return [error_doc], [metadata], None, None

    # Regular handlers for non-tabular files
    @staticmethod
    def _handle_pdf(pdf_path: str) -> tuple[List[Document], List[Dict]]:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            metadata = {
                "file_type": "PDF", 
                "source": pdf_path,
                "pages": len(docs)
            }
            return docs, [metadata]
        except Exception as e:
            metadata = {"file_type": "PDF", "source": pdf_path, "error": str(e)}
            doc = Document(
                page_content=f"PDF file at {pdf_path} could not be processed. Error: {str(e)}",
                metadata=metadata
            )
            return [doc], [metadata]

    @staticmethod
    def _handle_text(text_path: str) -> tuple[List[Document], List[Dict]]:
        try:
            loader = TextLoader(text_path, encoding='utf-8')
            docs = loader.load()
        except UnicodeDecodeError:
            loader = TextLoader(text_path, encoding='latin-1')
            docs = loader.load()
        
        metadata = {"file_type": "Text", "source": text_path}
        return docs, [metadata]

    @staticmethod
    def _handle_markdown(md_path: str) -> tuple[List[Document], List[Dict]]:
        loader = TextLoader(md_path, encoding='utf-8')
        docs = loader.load()
        metadata = {"file_type": "Markdown", "source": md_path}
        return docs, [metadata]

    @staticmethod
    def _handle_docx(docx_path: str) -> tuple[List[Document], List[Dict]]:
        loader = Docx2txtLoader(docx_path)
        docs = loader.load()
        metadata = {"file_type": "DOCX", "source": docx_path}
        return docs, [metadata]

class EnhancedAIAgentTools:
    """Enhanced AI agent tools with specialized tabular data capabilities"""
    
    def __init__(self, vectorstore, documents_metadata, dataframes, tabular_summaries):
        self.vectorstore = vectorstore
        self.documents_metadata = documents_metadata
        self.dataframes = dataframes
        self.tabular_summaries = tabular_summaries
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def search_documents(self, query: str) -> str:
        """Enhanced document search with tabular data awareness"""
        try:
            if not self.vectorstore:
                return "No documents are currently loaded."
            
            docs = self.vectorstore.similarity_search(query, k=5)
            if not docs:
                return "No relevant documents found for your query."
            
            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                file_name = os.path.basename(source)
                content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                
                # Add file type context
                file_type = doc.metadata.get('file_type', 'Unknown')
                if file_type in ['CSV', 'Excel']:
                    sheet_info = f" (Sheet: {doc.metadata.get('sheet_name', 'N/A')})" if doc.metadata.get('sheet_name') else ""
                    results.append(f"Result {i} from {file_name}{sheet_info} [{file_type}]:\n{content}")
                else:
                    results.append(f"Result {i} from {file_name} [{file_type}]:\n{content}")
            
            return "\n\n".join(results)
        except Exception as e:
            return f"Error searching documents: {str(e)}"

    def analyze_tabular_data(self, query: str) -> str:
        """Specialized analysis for tabular data"""
        try:
            if not self.dataframes:
                return "No tabular data (CSV/Excel) files are currently loaded."
            
            # Find relevant dataframes based on query
            relevant_data = []
            query_lower = query.lower()
            
            for file_path, df in self.dataframes.items():
                file_name = os.path.basename(file_path)
                summary = self.tabular_summaries.get(file_path, {})
                
                # Check if query matches column names or data content
                column_match = any(query_lower in col.lower() for col in df.columns)
                content_match = any(query_lower in str(val).lower() for val in df.values.flatten() if pd.notna(val))
                
                if column_match or content_match or query_lower in file_name.lower():
                    relevant_data.append((file_path, df, summary))
            
            if not relevant_data:
                return f"No tabular data found matching '{query}'. Available datasets: {list(self.dataframes.keys())}"
            
            # Analyze relevant data
            analysis_results = []
            for file_path, df, summary in relevant_data:
                file_name = os.path.basename(file_path)
                
                # Perform specific analysis based on query
                if any(keyword in query_lower for keyword in ['count', 'total', 'sum', 'number']):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        analysis_results.append(f"**{file_name} - Numeric Summary:**")
                        for col in numeric_cols:
                            total = df[col].sum()
                            count = df[col].count()
                            analysis_results.append(f"- {col}: Total = {total}, Count = {count}")
                
                elif any(keyword in query_lower for keyword in ['unique', 'distinct', 'categories']):
                    analysis_results.append(f"**{file_name} - Unique Values:**")
                    for col in df.columns:
                        unique_count = df[col].nunique()
                        analysis_results.append(f"- {col}: {unique_count} unique values")
                        if df[col].dtype == 'object' and unique_count <= 10:
                            unique_vals = df[col].dropna().unique()
                            analysis_results.append(f"  Values: {list(unique_vals)}")
                
                else:
                    # General analysis
                    analysis_results.append(f"**{file_name} Analysis:**")
                    analysis_results.append(f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    analysis_results.append(f"- Columns: {list(df.columns)}")
                    
                    # Show relevant data matching query
                    for col in df.columns:
                        if query_lower in col.lower():
                            value_counts = df[col].value_counts().head(5)
                            analysis_results.append(f"- {col} top values: {dict(value_counts)}")
            
            return "\n".join(analysis_results) if analysis_results else f"No specific analysis could be performed for '{query}'"
            
        except Exception as e:
            return f"Error analyzing tabular data: {str(e)}"

    def query_specific_data(self, query: str) -> str:
        """Query specific data points from tabular files"""
        try:
            if not self.dataframes:
                return "No tabular data available for querying."
            
            query_lower = query.lower()
            results = []
            
            for file_path, df in self.dataframes.items():
                file_name = os.path.basename(file_path)
                
                # Search in column names
                matching_cols = [col for col in df.columns if query_lower in col.lower()]
                
                if matching_cols:
                    results.append(f"**{file_name} - Columns matching '{query}':**")
                    for col in matching_cols:
                        # Show statistics for numeric columns
                        if df[col].dtype in ['int64', 'float64']:
                            stats = df[col].describe()
                            results.append(f"- {col}: Mean={stats['mean']:.2f}, Min={stats['min']}, Max={stats['max']}")
                        else:
                            # Show value counts for categorical
                            value_counts = df[col].value_counts().head(3)
                            results.append(f"- {col}: Top values = {dict(value_counts)}")
                
                # Search in data content
                text_cols = df.select_dtypes(include=['object']).columns
                for col in text_cols:
                    matching_rows = df[df[col].astype(str).str.contains(query, case=False, na=False)]
                    if not matching_rows.empty and len(matching_rows) <= 10:
                        results.append(f"**{file_name} - Rows containing '{query}' in {col}:**")
                        for idx, row in matching_rows.head(3).iterrows():
                            row_info = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
                            results.append(f"- Row {idx}: {row_info}")
            
            return "\n".join(results) if results else f"No specific data found for '{query}'"
            
        except Exception as e:
            return f"Error querying specific data: {str(e)}"

    def analyze_document_statistics(self, query: str = "") -> str:
        """Enhanced document statistics including tabular data"""
        try:
            if not self.documents_metadata:
                return "No documents metadata available."
            
            stats = {
                "total_documents": len(self.documents_metadata),
                "file_types": {},
                "pdf_pages": 0,
                "tabular_files": 0,
                "total_rows": 0,
                "total_columns": 0
            }
            
            for meta in self.documents_metadata:
                file_type = meta.get("file_type", "Unknown")
                stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
                
                if file_type == "PDF":
                    stats["pdf_pages"] += meta.get("pages", 0)
                elif file_type in ["CSV", "Excel"]:
                    stats["tabular_files"] += 1
                    stats["total_rows"] += meta.get("rows", 0)
                    stats["total_columns"] += meta.get("columns", 0)
            
            result = f"Document Statistics:\n"
            result += f"- Total Documents: {stats['total_documents']}\n"
            result += f"- File Types: {dict(stats['file_types'])}\n"
            if stats["pdf_pages"] > 0:
                result += f"- Total PDF Pages: {stats['pdf_pages']}\n"
            if stats["tabular_files"] > 0:
                result += f"- Tabular Files: {stats['tabular_files']}\n"
                result += f"- Total Data Rows: {stats['total_rows']}\n"
                result += f"- Average Columns per File: {stats['total_columns'] // max(stats['tabular_files'], 1)}\n"
            
            # Add tabular data details
            if self.dataframes:
                result += f"\nTabular Data Details:\n"
                for file_path, df in self.dataframes.items():
                    file_name = os.path.basename(file_path)
                    result += f"- {file_name}: {df.shape[0]} rows × {df.shape[1]} columns\n"
                    result += f"  Columns: {list(df.columns)}\n"
            
            return result
        except Exception as e:
            return f"Error analyzing document statistics: {str(e)}"

    def summarize_content(self, query: str) -> str:
        """Enhanced content summarization with tabular data awareness"""
        try:
            # Search for relevant documents
            relevant_docs = self.vectorstore.similarity_search(query, k=3)
            if not relevant_docs:
                return "No relevant content found to summarize."
            
            # Combine content from relevant documents
            combined_content = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Check if any tabular data is relevant
            tabular_context = ""
            if self.dataframes:
                query_lower = query.lower()
                for file_path, df in self.dataframes.items():
                    file_name = os.path.basename(file_path)
                    if any(query_lower in col.lower() for col in df.columns) or query_lower in file_name.lower():
                        tabular_context += f"\n\nTabular data from {file_name}:\n"
                        tabular_context += f"Columns: {list(df.columns)}\n"
                        tabular_context += f"Sample data:\n{df.head(2).to_string()}\n"
            
            # Use OpenAI to generate summary
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert document analyzer with specialization in both text and tabular data. Provide clear, concise summaries that highlight key insights from both text documents and structured data."},
                    {"role": "user", "content": f"Please summarize the following content related to '{query}':\n\n{combined_content[:2000]}{tabular_context}"}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

def create_enhanced_agent(vectorstore, documents_metadata, dataframes, tabular_summaries):
    """Create an enhanced AI agent with specialized tabular data tools"""
    try:
        # Initialize enhanced tools
        agent_tools = EnhancedAIAgentTools(vectorstore, documents_metadata, dataframes, tabular_summaries)
        
        # Define tools for the agent
        tools = [
            Tool(
                name="Search All Documents",
                func=agent_tools.search_documents,
                description="Search through all loaded documents (PDFs, text files, CSV, Excel) for specific information. Use this for general document search."
            ),
            Tool(
                name="Analyze Tabular Data",
                func=agent_tools.analyze_tabular_data,
                description="Specialized analysis of CSV and Excel files. Use this for questions about data analysis, statistics, column information, or specific data insights."
            ),
            Tool(
                name="Query Specific Data",
                func=agent_tools.query_specific_data,
                description="Query specific data points, values, or patterns from CSV/Excel files. Use this for finding specific records or data values."
            ),
            Tool(
                name="Document Statistics",
                func=agent_tools.analyze_document_statistics,
                description="Get comprehensive statistics about all loaded documents including file types, counts, and tabular data summaries."
            ),
            Tool(
                name="Summarize Content",
                func=agent_tools.summarize_content,
                description="Generate intelligent summaries of document content with awareness of both text and tabular data. Use this for creating comprehensive summaries."
            )
        ]
# Initialize LLM
        llm = ChatOpenAI(
            temperature=0.3,
            model_name="gpt-3.5-turbo",
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True
        )
        
        return agent
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return None

def process_data():
    """Process data from GitHub repository"""
    try:
        with st.spinner("Downloading and processing data from GitHub..."):
            # Download and extract files
            temp_dir = EnhancedDocumentProcessor.download_and_extract_zip(GITHUB_ZIP_URL)
            if not temp_dir:
                return False
            
            st.session_state.temp_dir = temp_dir
            
            # Load and process documents
            documents, metadata, dataframes, tabular_summaries = EnhancedDocumentProcessor.load_documents(temp_dir)
            
            if not documents:
                st.error("No documents were loaded successfully.")
                return False
            
            # Store processed data
            st.session_state.documents_metadata = metadata
            st.session_state.dataframes = dataframes
            st.session_state.tabular_data_summary = tabular_summaries
            
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            # Split documents
            split_docs = text_splitter.split_documents(documents)
            
            # Create embeddings and vectorstore
            embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            st.session_state.vectorstore = vectorstore
            
            # Create enhanced agent
            agent = create_enhanced_agent(vectorstore, metadata, dataframes, tabular_summaries)
            st.session_state.agent = agent
            
            st.success(f"Successfully processed {len(documents)} documents!")
            st.session_state.processed_data = True
            
            return True
            
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return False

def display_data_overview():
    """Display overview of processed data"""
    if not st.session_state.processed_data:
        return
    
    st.subheader("📊 Data Overview")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Document Summary", "Tabular Data", "File Details"])
    
    with tab1:
        if st.session_state.documents_metadata:
            # File type distribution
            file_types = {}
            total_files = len(st.session_state.documents_metadata)
            
            for meta in st.session_state.documents_metadata:
                file_type = meta.get("file_type", "Unknown")
                file_types[file_type] = file_types.get(file_type, 0) + 1
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Files", total_files)
                st.write("**File Types:**")
                for file_type, count in file_types.items():
                    st.write(f"- {file_type}: {count}")
            
            with col2:
                if st.session_state.dataframes:
                    total_rows = sum(df.shape[0] for df in st.session_state.dataframes.values())
                    st.metric("Total Data Rows", total_rows)
                    st.metric("Tabular Files", len(st.session_state.dataframes))
    
    with tab2:
        if st.session_state.dataframes:
            st.write("**Available Datasets:**")
            for file_path, df in st.session_state.dataframes.items():
                file_name = os.path.basename(file_path)
                with st.expander(f"📈 {file_name} ({df.shape[0]} rows × {df.shape[1]} columns)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Columns:**")
                        for col in df.columns:
                            st.write(f"- {col} ({df[col].dtype})")
                    
                    with col2:
                        st.write("**Sample Data:**")
                        st.dataframe(df.head(3))
                    
                    # Data summary
                    if file_path in st.session_state.tabular_data_summary:
                        summary = st.session_state.tabular_data_summary[file_path]
                        if "summary_stats" in summary and summary["summary_stats"]:
                            st.write("**Numeric Statistics:**")
                            stats_df = pd.DataFrame(summary["summary_stats"]).round(2)
                            st.dataframe(stats_df)
        else:
            st.info("No tabular data files found.")
    
    with tab3:
        if st.session_state.documents_metadata:
            st.write("**Detailed File Information:**")
            for i, meta in enumerate(st.session_state.documents_metadata, 1):
                source = meta.get("source", "Unknown")
                file_name = os.path.basename(source)
                file_type = meta.get("file_type", "Unknown")
                
                with st.expander(f"{i}. {file_name} [{file_type}]"):
                    st.json(meta)

def main():
    """Main application function"""
    # Header
    st.title("🤖 Kaizen Engineers AI Agent Knowledge Base")
    st.markdown("### Enhanced AI Assistant with Advanced Tabular Data Analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input("OpenAI API Key", type="password", 
                               value=os.environ.get("OPENAI_API_KEY", ""))
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        st.divider()
        
        # Data processing section
        st.header("Data Processing")
        
        if not st.session_state.processed_data:
            st.info("Click below to download and process data from GitHub repository.")
            
            if st.button("🚀 Initialize Knowledge Base", type="primary"):
                if not api_key:
                    st.error("Please provide OpenAI API Key first!")
                else:
                    st.session_state.initialization_started = True
                    if process_data():
                        st.rerun()
        else:
            st.success("✅ Knowledge base is ready!")
            if st.button("🔄 Refresh Data"):
                # Clear session state and reinitialize
                for key in ['agent', 'vectorstore', 'processed_data', 'documents_metadata', 
                           'dataframes', 'tabular_data_summary']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        st.divider()
        
        # System status
        st.header("System Status")
        status_items = [
            ("API Key", "✅" if api_key else "❌"),
            ("Data Processed", "✅" if st.session_state.processed_data else "❌"),
            ("Agent Ready", "✅" if st.session_state.agent else "❌"),
            ("Vector Store", "✅" if st.session_state.vectorstore else "❌")
        ]
        
        for item, status in status_items:
            st.write(f"{item}: {status}")
    
    # Main content area
    if not api_key:
        st.warning("⚠️ Please provide your OpenAI API Key in the sidebar to get started.")
        st.info("""
        This application requires an OpenAI API key to function. Your key is used to:
        - Create document embeddings for similarity search
        - Power the AI agent for intelligent responses
        - Generate summaries and analyses
        
        Your API key is not stored and is only used for the current session.
        """)
        return
    
    if not st.session_state.processed_data:
        st.info("👈 Please initialize the knowledge base using the sidebar to get started.")
        
        st.markdown("""
        ## Features
        
        This enhanced AI assistant provides:
        
        **📄 Document Processing:**
        - PDF, Text, Markdown, Word documents
        - Advanced CSV and Excel file analysis
        - Intelligent content extraction and indexing
        
        **🧠 AI Capabilities:**
        - Natural language querying across all documents
        - Specialized tabular data analysis
        - Statistical insights and data summaries
        - Context-aware responses
        
        **📊 Tabular Data Features:**
        - Comprehensive data profiling
        - Column-wise analysis and statistics
        - Pattern recognition and insights
        - Interactive data exploration
        """)
        return
    
    # Display data overview
    display_data_overview()
    
    # Chat interface
    st.divider()
    st.subheader("💬 Chat with Your AI Assistant")
    
    # Chat history display
    if st.session_state.chat_history:
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.chat_message("user").write(message)
            else:
                st.chat_message("assistant").write(message)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents and data..."):
        if not st.session_state.agent:
            st.error("AI Agent is not initialized. Please refresh the page and try again.")
            return
        
        # Display user message
        st.chat_message("user").write(prompt)
        st.session_state.chat_history.append(("user", prompt))
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.run(prompt)
                    st.write(response)
                    st.session_state.chat_history.append(("assistant", response))
                except Exception as e:
                    error_message = f"I encountered an error while processing your request: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_history.append(("assistant", error_message))
    
    # Example queries
    if st.session_state.processed_data:
        st.divider()
        st.subheader("💡 Example Queries")
        
        example_queries = [
            "What documents do you have access to?",
            "Show me statistics about the data",
            "What columns are in the CSV files?",
            "Analyze the numerical data in the datasets",
            "Find information about orders or sales",
            "Summarize the content of all documents"
        ]
        
        cols = st.columns(2)
        for i, query in enumerate(example_queries):
            col = cols[i % 2]
            if col.button(query, key=f"example_{i}"):
                # Simulate clicking the example query
                st.session_state.chat_history.append(("user", query))
                with st.spinner("Processing example query..."):
                    try:
                        response = st.session_state.agent.run(query)
                        st.session_state.chat_history.append(("assistant", response))
                        st.rerun()
                    except Exception as e:
                        error_message = f"Error processing example query: {str(e)}"
                        st.session_state.chat_history.append(("assistant", error_message))
                        st.rerun()

# Cleanup function
def cleanup_temp_files():
    """Clean up temporary files on app shutdown"""
    if st.session_state.get("temp_dir") and os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir)
        except:
            pass

# Register cleanup
import atexit
atexit.register(cleanup_temp_files)

if __name__ == "__main__":
    main()
