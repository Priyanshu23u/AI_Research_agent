import pandas as pd
from typing import Dict, List
from .enhanced_analyzer import PaperAnalysis, ResearchGaps
from .generator import LocalGenerator
from .paper_builder import assemble_markdown, write_docx, write_pdf

class EnhancedPaperBuilder:
    """Enhanced paper builder with analysis-driven generation"""
    
    def __init__(self, generator: LocalGenerator):
        self.generator = generator
    
    def generate_enhanced_paper(self, topic: str, analyses: List[PaperAnalysis], 
                              gaps: ResearchGaps, trends: Dict) -> Dict[str, str]:
        """Generate research paper using comprehensive analysis"""
        sections = {}
        
        # Enhanced Abstract
        sections["Abstract"] = self._generate_abstract(topic, analyses, trends)
        
        # Enhanced Introduction
        sections["Introduction"] = self._generate_introduction(topic, trends)
        
        # Enhanced Literature Review with synthesis
        sections["Literature Review"] = self._generate_literature_review(analyses)
        
        # Research Gaps Section
        sections["Research Gaps"] = self._generate_research_gaps(gaps)
        
        # Key Findings with synthesis
        sections["Key Findings"] = self._generate_key_findings(analyses)
        
        # Enhanced Results & Discussion
        sections["Results & Discussion"] = self._generate_results_discussion(analyses, trends)
        
        # Enhanced Conclusion
        sections["Conclusion"] = self._generate_conclusion(topic, gaps)
        
        # Enhanced Future Work
        sections["Future Work"] = self._generate_future_work(gaps, analyses)
        
        return sections
    
    def _generate_abstract(self, topic: str, analyses: List[PaperAnalysis], trends: Dict) -> str:
        """Generate enhanced abstract with synthesis"""
        prompt = f"""
Write a comprehensive abstract for a survey paper on "{topic}".

Based on analysis of {len(analyses)} papers:
- Average quality score: {trends.get('average_quality_score', 'N/A')}
- Top research domains: {', '.join([d[0] for d in trends.get('top_domains', [])])}
- High-quality papers: {trends.get('high_quality_papers', 0)}

The abstract should:
- Define the research area and its importance
- Summarize key findings and trends
- Highlight research gaps and future directions
- Be 250-300 words in formal academic style

Abstract:
"""
        return self.generator.generate(prompt, max_new_tokens=400, temperature=0.4)
    
    def _generate_literature_review(self, analyses: List[PaperAnalysis]) -> str:
        """Generate synthesized literature review"""
        # Group papers by methodology
        method_groups = {}
        for analysis in analyses:
            if analysis.methodology:
                method = analysis.methodology[:50]  # Truncate for grouping
                if method not in method_groups:
                    method_groups[method] = []
                method_groups[method].append(analysis)
        
        review_sections = []
        for method, papers in method_groups.items():
            if len(papers) >= 2:  # Only include methods with multiple papers
                section = f"### {method}\n\n"
                paper_summaries = []
                for paper in papers[:3]:  # Limit to top 3 papers per method
                    contributions = '. '.join(paper.key_contributions[:2])
                    paper_summaries.append(f"{paper.title}: {contributions}")
                
                section += f"Multiple studies have explored {method}. "
                section += ' '.join(paper_summaries[:2])  # Limit for context
                review_sections.append(section)
        
        return '\n\n'.join(review_sections)
    
    def _generate_research_gaps(self, gaps: ResearchGaps) -> str:
        """Generate research gaps section"""
        prompt = f"""
Write a comprehensive Research Gaps section based on identified gaps:

METHODOLOGY GAPS:
{chr(10).join(gaps.methodology_gaps)}

DATASET GAPS:
{chr(10).join(gaps.dataset_gaps)}

EVALUATION GAPS:
{chr(10).join(gaps.evaluation_gaps)}

APPLICATION GAPS:
{chr(10).join(gaps.application_gaps)}

THEORETICAL GAPS:
{chr(10).join(gaps.theoretical_gaps)}

Structure the section with subsections for each gap type. Explain why each gap exists and its importance.
"""
        return self.generator.generate(prompt, max_new_tokens=1000, temperature=0.4)
    
    def _generate_key_findings(self, analyses: List[PaperAnalysis]) -> str:
        """Generate synthesized key findings"""
        all_contributions = []
        for analysis in analyses:
            all_contributions.extend(analysis.key_contributions)
        
        # Get top quality papers
        top_papers = sorted(analyses, key=lambda x: x.quality_score, reverse=True)[:5]
        
        findings_text = "## Major Contributions\n\n"
        for i, paper in enumerate(top_papers, 1):
            if paper.key_contributions:
                findings_text += f"{i}. **{paper.title}**: {'. '.join(paper.key_contributions[:2])}\n\n"
        
        return findings_text
    
    def _generate_results_discussion(self, analyses: List[PaperAnalysis], trends: Dict) -> str:
        """Generate results and discussion with trend analysis"""
        prompt = f"""
Write a Results & Discussion section based on:

RESEARCH TRENDS:
- Total papers analyzed: {trends.get('total_papers', 0)}
- Average quality score: {trends.get('average_quality_score', 'N/A')}
- High-quality papers: {trends.get('high_quality_papers', 0)}
- Novel approaches: {trends.get('novel_papers', 0)}

TOP METHODOLOGIES:
{chr(10).join([f"- {m[0]}: {m[1]} papers" for m in trends.get('top_methodologies', [])])}

Discuss the state of the field, methodology trends, quality patterns, and research evolution.
"""
        return self.generator.generate(prompt, max_new_tokens=800, temperature=0.4)
    
    def _generate_conclusion(self, topic: str, gaps: ResearchGaps) -> str:
        """Generate conclusion with identified gaps"""
        prompt = f"""
Write a comprehensive conclusion for the survey on "{topic}".

Key research gaps identified:
- Methodology gaps: {len(gaps.methodology_gaps)} identified
- Dataset gaps: {len(gaps.dataset_gaps)} identified  
- Evaluation gaps: {len(gaps.evaluation_gaps)} identified
- Application gaps: {len(gaps.application_gaps)} identified

Summarize the current state, key insights, and most critical gaps for future research.
"""
        return self.generator.generate(prompt, max_new_tokens=600, temperature=0.4)
    
    def _generate_future_work(self, gaps: ResearchGaps, analyses: List[PaperAnalysis]) -> str:
        """Generate future work based on gaps and analysis"""
        future_suggestions = []
        for analysis in analyses:
            future_suggestions.extend(analysis.future_work)
        
        prompt = f"""
Propose concrete future work directions based on:

IDENTIFIED GAPS:
- Methodology: {', '.join(gaps.methodology_gaps[:3])}
- Datasets: {', '.join(gaps.dataset_gaps[:3])}
- Applications: {', '.join(gaps.application_gaps[:3])}

RESEARCHER SUGGESTIONS:
{chr(10).join(future_suggestions[:10])}

Provide 5-7 specific, actionable future research directions with justification.
"""
        return self.generator.generate(prompt, max_new_tokens=800, temperature=0.5)
