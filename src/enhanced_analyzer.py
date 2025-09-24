import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import re
from dataclasses import dataclass
from .generator import LocalGenerator

@dataclass
class PaperAnalysis:
    """Structured analysis results for a paper"""
    paper_id: str
    title: str
    key_contributions: List[str]
    methodology: str
    datasets_used: List[str]
    metrics: List[str]
    limitations: List[str]
    future_work: List[str]
    research_domain: str
    quality_score: float
    novelty_score: float

@dataclass
class ResearchGaps:
    """Identified research gaps and opportunities"""
    methodology_gaps: List[str]
    dataset_gaps: List[str]
    evaluation_gaps: List[str]
    application_gaps: List[str]
    theoretical_gaps: List[str]

class EnhancedPaperAnalyzer:
    """Advanced paper analysis with LLM-powered insights"""
    
    def __init__(self, generator: LocalGenerator):
        self.generator = generator
        self.analysis_cache = {}
        
    def analyze_paper(self, paper: Dict[str, Any]) -> PaperAnalysis:
        """Analyze a single paper comprehensively"""
        paper_id = paper.get('id', '')
        
        if paper_id in self.analysis_cache:
            return self.analysis_cache[paper_id]
        
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        
        analysis_prompt = f"""
Analyze this research paper and extract key information:

Title: {title}
Abstract: {abstract}

Extract the following information:

1. KEY CONTRIBUTIONS (3-5 main contributions):
2. METHODOLOGY (main approach/method used):
3. DATASETS (datasets mentioned):
4. METRICS (evaluation metrics used):
5. LIMITATIONS (acknowledged limitations):
6. FUTURE WORK (suggested future directions):
7. RESEARCH DOMAIN (primary research area):
8. QUALITY SCORE (1-10, based on novelty, rigor, impact):
9. NOVELTY SCORE (1-10, how novel/innovative):

Format response with numbered sections exactly as shown.
"""
        
        try:
            response = self.generator.generate(analysis_prompt, max_new_tokens=800, temperature=0.3)
            analysis = self._parse_analysis_response(response, paper_id, title)
            self.analysis_cache[paper_id] = analysis
            return analysis
        except Exception as e:
            print(f"Failed to analyze paper {paper_id}: {e}")
            return self._create_default_analysis(paper_id, title)
    
    def _parse_analysis_response(self, response: str, paper_id: str, title: str) -> PaperAnalysis:
        """Parse LLM response into structured analysis"""
        sections = {}
        current_section = None
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if any(keyword in line.upper() for keyword in [
                'KEY CONTRIBUTIONS', 'METHODOLOGY', 'DATASETS', 'METRICS',
                'LIMITATIONS', 'FUTURE WORK', 'RESEARCH DOMAIN', 'QUALITY SCORE', 'NOVELTY SCORE'
            ]):
                if 'KEY CONTRIBUTIONS' in line.upper():
                    current_section = 'contributions'
                elif 'METHODOLOGY' in line.upper():
                    current_section = 'methodology'
                elif 'DATASETS' in line.upper():
                    current_section = 'datasets'
                elif 'METRICS' in line.upper():
                    current_section = 'metrics'
                elif 'LIMITATIONS' in line.upper():
                    current_section = 'limitations'
                elif 'FUTURE WORK' in line.upper():
                    current_section = 'future_work'
                elif 'RESEARCH DOMAIN' in line.upper():
                    current_section = 'domain'
                elif 'QUALITY SCORE' in line.upper():
                    current_section = 'quality'
                elif 'NOVELTY SCORE' in line.upper():
                    current_section = 'novelty'
                sections[current_section] = []
            elif current_section and line:
                sections[current_section].append(line)
        
        quality_score = self._extract_score(sections.get('quality', ['5.0']))
        novelty_score = self._extract_score(sections.get('novelty', ['5.0']))
        
        return PaperAnalysis(
            paper_id=paper_id,
            title=title,
            key_contributions=sections.get('contributions', []),
            methodology=' '.join(sections.get('methodology', [])),
            datasets_used=sections.get('datasets', []),
            metrics=sections.get('metrics', []),
            limitations=sections.get('limitations', []),
            future_work=sections.get('future_work', []),
            research_domain=' '.join(sections.get('domain', [])),
            quality_score=quality_score,
            novelty_score=novelty_score
        )
    
    def _extract_score(self, score_lines: List[str]) -> float:
        """Extract numeric score from text"""
        for line in score_lines:
            numbers = re.findall(r'\d+\.?\d*', line)
            if numbers:
                try:
                    score = float(numbers[0])
                    return min(10.0, max(1.0, score))
                except ValueError:
                    continue
        return 5.0
    
    def _create_default_analysis(self, paper_id: str, title: str) -> PaperAnalysis:
        """Create default analysis when LLM analysis fails"""
        return PaperAnalysis(
            paper_id=paper_id, title=title, key_contributions=[], methodology="",
            datasets_used=[], metrics=[], limitations=[], future_work=[],
            research_domain="", quality_score=5.0, novelty_score=5.0
        )
    
    def analyze_papers_batch(self, papers: List[Dict[str, Any]]) -> List[PaperAnalysis]:
        """Analyze multiple papers with progress tracking"""
        analyses = []
        print(f"Analyzing {len(papers)} papers...")
        for i, paper in enumerate(papers):
            print(f"Analyzing paper {i+1}/{len(papers)}: {paper.get('title', '')[:80]}...")
            analysis = self.analyze_paper(paper)
            analyses.append(analysis)
        return analyses
    
    def identify_research_gaps(self, analyses: List[PaperAnalysis]) -> ResearchGaps:
        """Identify research gaps from analyzed papers"""
        methodologies = [a.methodology for a in analyses if a.methodology]
        datasets = []
        for a in analyses:
            datasets.extend(a.datasets_used)
        
        gap_prompt = f"""
Based on research analysis, identify key gaps:

METHODOLOGIES: {', '.join(set(methodologies[:10]))}
DATASETS: {', '.join(set(datasets[:10]))}

Identify gaps in:
1. METHODOLOGY GAPS (missing approaches)
2. DATASET GAPS (missing data types)  
3. EVALUATION GAPS (missing metrics)
4. APPLICATION GAPS (underexplored uses)
5. THEORETICAL GAPS (theory gaps)

List 3-5 gaps per category.
"""
        
        try:
            response = self.generator.generate(gap_prompt, max_new_tokens=1000, temperature=0.4)
            return self._parse_gaps_response(response)
        except Exception:
            return ResearchGaps([], [], [], [], [])
    
    def _parse_gaps_response(self, response: str) -> ResearchGaps:
        """Parse research gaps from LLM response"""
        gaps = ResearchGaps([], [], [], [], [])
        current_category = None
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if 'METHODOLOGY GAPS' in line.upper():
                current_category = 'methodology'
            elif 'DATASET GAPS' in line.upper():
                current_category = 'dataset'
            elif 'EVALUATION GAPS' in line.upper():
                current_category = 'evaluation'
            elif 'APPLICATION GAPS' in line.upper():
                current_category = 'application'
            elif 'THEORETICAL GAPS' in line.upper():
                current_category = 'theoretical'
            elif current_category and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                clean_line = re.sub(r'^[-•\d\s\.\)]+', '', line).strip()
                if clean_line:
                    if current_category == 'methodology':
                        gaps.methodology_gaps.append(clean_line)
                    elif current_category == 'dataset':
                        gaps.dataset_gaps.append(clean_line)
                    elif current_category == 'evaluation':
                        gaps.evaluation_gaps.append(clean_line)
                    elif current_category == 'application':
                        gaps.application_gaps.append(clean_line)
                    elif current_category == 'theoretical':
                        gaps.theoretical_gaps.append(clean_line)
        return gaps
