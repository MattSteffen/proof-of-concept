#!/usr/bin/env python3
"""
Perspective-aware document processing pipeline
Run with: uv run python scripts/create_training_data.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re
from dataclasses import dataclass
from pathlib import Path

# from sec_edgar_downloader import Downloader


@dataclass
class PerspectiveQuery:
    query: str
    perspective: str
    document_id: str
    relevance_score: int  # 0-3 scale
    reasoning: str


class DocumentPerspectiveGenerator:
    def __init__(self):
        self.perspectives = {
            "risk_analyst": {
                "focus": "financial risks, regulatory issues, market threats",
                "keywords": [
                    "decline",
                    "loss",
                    "risk",
                    "litigation",
                    "regulatory",
                    "competition",
                ],
                "query_templates": [
                    "What are the key financial risks facing {company}?",
                    "regulatory compliance issues for {company}",
                    "market competition threats to {company}",
                    "liquidity and debt concerns for {company}",
                    "operational risks in {company} business",
                ],
            },
            "growth_analyst": {
                "focus": "expansion opportunities, revenue growth, market potential",
                "keywords": [
                    "growth",
                    "expansion",
                    "opportunity",
                    "revenue",
                    "market",
                    "innovation",
                ],
                "query_templates": [
                    "What growth opportunities does {company} have?",
                    "revenue expansion strategies for {company}",
                    "new market opportunities for {company}",
                    "innovation and R&D investments at {company}",
                    "acquisition and partnership opportunities for {company}",
                ],
            },
            "operations_analyst": {
                "focus": "efficiency, cost management, operational improvements",
                "keywords": [
                    "efficiency",
                    "cost",
                    "operations",
                    "margin",
                    "productivity",
                    "process",
                ],
                "query_templates": [
                    "How can {company} improve operational efficiency?",
                    "cost reduction opportunities at {company}",
                    "operational margin improvements for {company}",
                    "supply chain optimization for {company}",
                    "workforce productivity at {company}",
                ],
            },
            "compliance_analyst": {
                "focus": "regulatory compliance, legal issues, governance",
                "keywords": [
                    "compliance",
                    "regulation",
                    "legal",
                    "governance",
                    "audit",
                    "ethics",
                ],
                "query_templates": [
                    "What compliance requirements does {company} face?",
                    "regulatory changes affecting {company}",
                    "governance and ethics issues at {company}",
                    "audit findings and compliance status for {company}",
                    "legal proceedings involving {company}",
                ],
            },
        }

    def download_sec_documents(
        self, ticker: str, form_type: str = "10-K", limit: int = 5
    ) -> List[str]:
        # """Download SEC documents for a given ticker"""
        # # Create data directory
        # data_dir = Path("data")
        # data_dir.mkdir(exist_ok=True)

        # dl = Downloader("MVP Research", "research@company.com")
        # dl.get(form_type, ticker, limit=limit)

        # # Return actual file paths
        # filing_dir = Path(f"sec-edgar-filings/{ticker}/{form_type}")
        # if filing_dir.exists():
        #     return list(filing_dir.glob("*.txt"))
        return []

    def extract_document_sections(self, document_text: str) -> Dict[str, str]:
        """Extract key sections from SEC filing"""
        sections = {}

        # Common SEC filing sections
        section_patterns = {
            "business_overview": r"Item 1\.\s*Business",
            "risk_factors": r"Item 1A\.\s*Risk Factors",
            "financial_data": r"Item 8\.\s*Financial Statements",
            "management_discussion": r"Item 7\.\s*Management.s Discussion",
            "controls": r"Item 9A\.\s*Controls and Procedures",
        }

        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                start_pos = match.start()
                # Extract next 2000 characters as sample
                sections[section_name] = document_text[start_pos : start_pos + 2000]

        return sections

    def generate_perspective_queries(
        self, company_name: str, document_sections: Dict[str, str]
    ) -> List[PerspectiveQuery]:
        """Generate queries from different analyst perspectives"""
        queries = []

        for perspective, config in self.perspectives.items():
            for template in config["query_templates"]:
                query = template.format(company=company_name)

                # Create query object (relevance to be labeled manually)
                perspective_query = PerspectiveQuery(
                    query=query,
                    perspective=perspective,
                    document_id=f"{company_name}_10K",
                    relevance_score=0,  # To be labeled
                    reasoning="",  # To be filled during labeling
                )
                queries.append(perspective_query)

        return queries

    def create_labeling_interface(
        self, queries: List[PerspectiveQuery], document_sections: Dict[str, str]
    ) -> pd.DataFrame:
        """Create a DataFrame for manual labeling"""
        labeling_data = []

        for query in queries:
            labeling_data.append(
                {
                    "query": query.query,
                    "perspective": query.perspective,
                    "document_id": query.document_id,
                    "relevance_score": query.relevance_score,
                    "reasoning": query.reasoning,
                    "document_preview": list(document_sections.values())[0][:200]
                    + "...",
                }
            )

        return pd.DataFrame(labeling_data)


# Relevance Scoring Guidelines
RELEVANCE_SCALE = {
    0: "Not relevant - Query perspective doesn't match document content",
    1: "Slightly relevant - Document has minimal information for this perspective",
    2: "Moderately relevant - Document has some useful information for this perspective",
    3: "Highly relevant - Document directly addresses this perspective with detailed information",
}


def label_relevance_batch(
    df: pd.DataFrame, start_idx: int = 0, batch_size: int = 10
) -> pd.DataFrame:
    """Interactive labeling function for batch processing"""
    print("RELEVANCE SCORING SCALE:")
    for score, description in RELEVANCE_SCALE.items():
        print(f"{score}: {description}")
    print("\n" + "=" * 50 + "\n")

    end_idx = min(start_idx + batch_size, len(df))

    for i in range(start_idx, end_idx):
        row = df.iloc[i]

        print(f"QUERY {i+1}/{len(df)}: {row['query']}")
        print(f"PERSPECTIVE: {row['perspective']}")
        print(f"DOCUMENT PREVIEW: {row['document_preview']}")
        print("-" * 30)

        # Get relevance score
        while True:
            try:
                score = int(input("Relevance Score (0-3): "))
                if 0 <= score <= 3:
                    df.at[i, "relevance_score"] = score
                    break
                else:
                    print("Please enter a score between 0-3")
            except ValueError:
                print("Please enter a valid number")

        # Get reasoning
        reasoning = input("Brief reasoning (optional): ")
        df.at[i, "reasoning"] = reasoning

        print("\n" + "=" * 50 + "\n")

    return df


# Example usage workflow
def main():
    # Initialize pipeline
    generator = DocumentPerspectiveGenerator()

    # 1. Download documents
    print("Step 1: Download SEC documents")
    # documents = generator.download_sec_documents("AAPL", "10-K", 2)

    # 2. For demo, use sample text
    sample_doc = """
    Apple Inc. faces significant risks from international operations, including currency fluctuations
    and geopolitical tensions. Revenue growth has been driven by iPhone sales and services expansion.
    The company continues to invest in R&D for new product categories. Operational efficiency has
    improved through supply chain optimization and manufacturing process improvements.
    """

    sections = {"business_overview": sample_doc}

    # 3. Generate perspective queries
    print("Step 2: Generate perspective queries")
    queries = generator.generate_perspective_queries("Apple Inc.", sections)

    # 4. Create labeling interface
    print("Step 3: Create labeling dataset")
    labeling_df = generator.create_labeling_interface(queries, sections)

    # 5. Label first batch
    print("Step 4: Begin labeling process")
    labeled_df = label_relevance_batch(labeling_df, batch_size=5)

    # 6. Save results
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "labeled_perspective_data.csv"

    labeled_df.to_csv(output_file, index=False)
    print(f"Labeled data saved to '{output_file}'")

    return labeled_df


if __name__ == "__main__":
    df = main()
    print(f"\nGenerated {len(df)} perspective queries for labeling")
    print("Next steps:")
    print("1. Complete labeling all queries")
    print("2. Add more companies/documents")
    print("3. Use labeled data to train siamese network")
