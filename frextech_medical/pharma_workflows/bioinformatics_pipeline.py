#!/usr/bin/env python3
"""
Bioinformatics Pipeline for FrexTech Medical
Genomics, proteomics, and multi-omics data integration
"""

import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import aiohttp
from pathlib import Path
import hashlib
import gzip
import pickle
from collections import defaultdict
import warnings

# Bioinformatics libraries
try:
    import biopython
    from Bio import SeqIO, Entrez, Seq, SeqRecord
    from Bio.Blast import NCBIWWW, NCBIXML
    from Bio.Align import MultipleSeqAlignment
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
except ImportError:
    warnings.warn("Biopython not installed. Some features may be limited.")
    
try:
    import pysam
except ImportError:
    warnings.warn("pysam not installed. BAM/CRAM support disabled.")
    
try:
    import vcf
except ImportError:
    warnings.warn("PyVCF not installed. VCF support disabled.")

# ML/Stats libraries
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import umap
import plotly.graph_objects as go
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OmicsDataType(Enum):
    """Types of omics data"""
    GENOMIC = "genomic"
    TRANSCRIPTOMIC = "transcriptomic"
    PROTEOMIC = "proteomic"
    METABOLOMIC = "metabolomic"
    EPIGENOMIC = "epigenomic"
    MULTIOMICS = "multiomics"

class DataSource(Enum):
    """Data source types"""
    NCBI = "ncbi"
    ENSEMBL = "ensembl"
    UNIPROT = "uniprot"
    TCGA = "tcga"
    GEO = "geo"
    LOCAL = "local"
    CUSTOM = "custom"

@dataclass
class SequenceData:
    """Container for sequence data"""
    sequence_id: str
    sequence: str
    sequence_type: str = "DNA"  # DNA, RNA, PROTEIN
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_scores: Optional[List[int]] = None
    
    def gc_content(self) -> float:
        """Calculate GC content percentage"""
        if not self.sequence:
            return 0.0
        gc_count = self.sequence.upper().count('G') + self.sequence.upper().count('C')
        return (gc_count / len(self.sequence)) * 100
    
    def translate(self, table: int = 1) -> str:
        """Translate DNA/RNA to protein"""
        if self.sequence_type not in ["DNA", "RNA"]:
            raise ValueError("Can only translate DNA or RNA sequences")
        
        seq_obj = Seq.Seq(self.sequence)
        if self.sequence_type == "RNA":
            seq_obj = seq_obj.transcribe()
        
        return str(seq_obj.translate(table=table))
    
    def reverse_complement(self) -> str:
        """Get reverse complement"""
        if self.sequence_type != "DNA":
            raise ValueError("Reverse complement only for DNA")
        
        seq_obj = Seq.Seq(self.sequence)
        return str(seq_obj.reverse_complement())

@dataclass
class VariantData:
    """Container for genetic variant data"""
    chromosome: str
    position: int
    reference: str
    alternate: str
    variant_id: Optional[str] = None
    quality: Optional[float] = None
    filter: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)
    samples: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def is_snv(self) -> bool:
        """Check if variant is single nucleotide variant"""
        return (len(self.reference) == 1 and 
                len(self.alternate) == 1 and 
                self.reference != self.alternate)
    
    def is_indel(self) -> bool:
        """Check if variant is insertion/deletion"""
        return len(self.reference) != len(self.alternate)
    
    def vcf_format(self) -> str:
        """Format as VCF line"""
        info_str = ";".join([f"{k}={v}" for k, v in self.info.items()]) if self.info else "."
        return f"{self.chromosome}\t{self.position}\t{self.variant_id or '.'}\t{self.reference}\t{self.alternate}\t{self.quality or '.'}\t{self.filter or '.'}\t{info_str}"

class BioinformaticsPipeline:
    """Main bioinformatics pipeline class"""
    
    def __init__(self, cache_dir: str = "./bioinformatics_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.sequences: Dict[str, SequenceData] = {}
        self.variants: List[VariantData] = []
        self.expression_data: pd.DataFrame = pd.DataFrame()
        self.protein_data: pd.DataFrame = pd.DataFrame()
        self.metadata: Dict[str, Any] = {}
        
        # Analysis results cache
        self.analysis_cache: Dict[str, Any] = {}
        
        # API clients
        self.ncbi_email = None  # Required for NCBI queries
        self.ncbi_api_key = None
        
    def configure_ncbi(self, email: str, api_key: Optional[str] = None):
        """Configure NCBI Entrez access"""
        self.ncbi_email = email
        self.ncbi_api_key = api_key
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
    
    async def fetch_sequence(self, accession_id: str, 
                           source: DataSource = DataSource.NCBI) -> Optional[SequenceData]:
        """Fetch sequence from remote database"""
        
        cache_key = f"sequence_{accession_id}_{source.value}"
        if cache_key in self.analysis_cache:
            logger.info(f"Using cached sequence for {accession_id}")
            return self.analysis_cache[cache_key]
        
        try:
            if source == DataSource.NCBI:
                return await self._fetch_from_ncbi(accession_id)
            elif source == DataSource.ENSEMBL:
                return await self._fetch_from_ensembl(accession_id)
            elif source == DataSource.UNIPROT:
                return await self._fetch_from_uniprot(accession_id)
            else:
                raise ValueError(f"Unsupported source: {source}")
                
        except Exception as e:
            logger.error(f"Error fetching sequence {accession_id}: {e}")
            return None
    
    async def _fetch_from_ncbi(self, accession_id: str) -> Optional[SequenceData]:
        """Fetch sequence from NCBI"""
        try:
            # Use Entrez to fetch sequence
            handle = Entrez.efetch(db="nucleotide", id=accession_id, 
                                  rettype="fasta", retmode="text")
            record = SeqIO.read(handle, "fasta")
            handle.close()
            
            sequence_data = SequenceData(
                sequence_id=record.id,
                sequence=str(record.seq),
                description=record.description,
                metadata={"source": "NCBI", "length": len(record.seq)}
            )
            
            # Cache the result
            cache_key = f"sequence_{accession_id}_ncbi"
            self.analysis_cache[cache_key] = sequence_data
            
            return sequence_data
            
        except Exception as e:
            logger.error(f"NCBI fetch error: {e}")
            return None
    
    async def _fetch_from_ensembl(self, accession_id: str) -> Optional[SequenceData]:
        """Fetch sequence from Ensembl"""
        async with aiohttp.ClientSession() as session:
            try:
                # Try REST API
                url = f"https://rest.ensembl.org/sequence/id/{accession_id}"
                headers = {"Content-Type": "application/json"}
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        sequence_data = SequenceData(
                            sequence_id=data.get("id", accession_id),
                            sequence=data.get("seq", ""),
                            description=data.get("desc", ""),
                            metadata={
                                "source": "Ensembl",
                                "species": data.get("species", ""),
                                "assembly": data.get("assembly", "")
                            }
                        )
                        
                        cache_key = f"sequence_{accession_id}_ensembl"
                        self.analysis_cache[cache_key] = sequence_data
                        
                        return sequence_data
                        
            except Exception as e:
                logger.error(f"Ensembl fetch error: {e}")
                
        return None
    
    async def _fetch_from_uniprot(self, accession_id: str) -> Optional[SequenceData]:
        """Fetch protein sequence from UniProt"""
        async with aiohttp.ClientSession() as session:
            try:
                url = f"https://www.uniprot.org/uniprot/{accession_id}.fasta"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        fasta_text = await response.text()
                        lines = fasta_text.strip().split('\n')
                        
                        header = lines[0]
                        sequence = ''.join(lines[1:])
                        
                        sequence_data = SequenceData(
                            sequence_id=accession_id,
                            sequence=sequence,
                            sequence_type="PROTEIN",
                            description=header[1:],  # Remove '>'
                            metadata={"source": "UniProt"}
                        )
                        
                        cache_key = f"sequence_{accession_id}_uniprot"
                        self.analysis_cache[cache_key] = sequence_data
                        
                        return sequence_data
                        
            except Exception as e:
                logger.error(f"UniProt fetch error: {e}")
                
        return None
    
    def load_vcf(self, vcf_file: str) -> List[VariantData]:
        """Load variants from VCF file"""
        variants = []
        
        try:
            import vcf
            vcf_reader = vcf.Reader(filename=vcf_file)
            
            for record in vcf_reader:
                variant = VariantData(
                    chromosome=record.CHROM,
                    position=record.POS,
                    reference=record.REF,
                    alternate=str(record.ALT[0]) if record.ALT else "",
                    variant_id=record.ID,
                    quality=record.QUAL,
                    filter=record.FILTER[0] if record.FILTER else "PASS",
                    info=dict(record.INFO),
                    samples={sample: dict(record.samples[sample]) for sample in record.samples}
                )
                variants.append(variant)
                
            self.variants.extend(variants)
            logger.info(f"Loaded {len(variants)} variants from {vcf_file}")
            
        except ImportError:
            logger.warning("PyVCF not installed, using simple parser")
            # Simple VCF parser
            with open(vcf_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 8:
                        variant = VariantData(
                            chromosome=parts[0],
                            position=int(parts[1]),
                            reference=parts[3],
                            alternate=parts[4],
                            variant_id=parts[2] if parts[2] != '.' else None,
                            quality=float(parts[5]) if parts[5] != '.' else None,
                            filter=parts[6],
                            info={}  # Simplified
                        )
                        variants.append(variant)
            
            self.variants.extend(variants)
            logger.info(f"Loaded {len(variants)} variants (simple parse) from {vcf_file}")
        
        return variants
    
    def load_expression_matrix(self, filepath: str, 
                             format: str = "csv") -> pd.DataFrame:
        """Load gene expression matrix"""
        if format == "csv":
            df = pd.read_csv(filepath, index_col=0)
        elif format == "tsv":
            df = pd.read_csv(filepath, sep='\t', index_col=0)
        elif format == "excel":
            df = pd.read_excel(filepath, index_col=0)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.expression_data = df
        logger.info(f"Loaded expression matrix: {df.shape[0]} genes x {df.shape[1]} samples")
        return df
    
    def differential_expression(self, group_a: List[str], group_b: List[str],
                              method: str = "t-test",
                              pvalue_threshold: float = 0.05,
                              log2fc_threshold: float = 1.0) -> pd.DataFrame:
        """Perform differential expression analysis"""
        
        if self.expression_data.empty:
            raise ValueError("No expression data loaded")
        
        # Ensure all samples exist
        all_samples = list(self.expression_data.columns)
        missing_a = [s for s in group_a if s not in all_samples]
        missing_b = [s for s in group_b if s not in all_samples]
        
        if missing_a or missing_b:
            raise ValueError(f"Missing samples: A={missing_a}, B={missing_b}")
        
        # Calculate statistics
        results = []
        
        for gene in self.expression_data.index:
            values_a = self.expression_data.loc[gene, group_a].values.astype(float)
            values_b = self.expression_data.loc[gene, group_b].values.astype(float)
            
            # Skip genes with missing values
            if np.any(np.isnan(values_a)) or np.any(np.isnan(values_b)):
                continue
            
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)
            
            # Avoid division by zero
            if mean_a == 0 or mean_b == 0:
                log2fc = 0
            else:
                log2fc = np.log2(mean_b / mean_a)
            
            # Statistical test
            if method == "t-test":
                from scipy import stats
                try:
                    t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
                except:
                    t_stat, p_value = np.nan, np.nan
            elif method == "mann-whitney":
                from scipy import stats
                try:
                    u_stat, p_value = stats.mannwhitneyu(values_a, values_b)
                except:
                    u_stat, p_value = np.nan, np.nan
            else:
                p_value = np.nan
            
            # Calculate fold change
            fold_change = mean_b / mean_a if mean_a != 0 else np.inf
            
            results.append({
                "gene": gene,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "fold_change": fold_change,
                "log2_fold_change": log2fc,
                "p_value": p_value,
                "significant": (abs(log2fc) >= log2fc_threshold and 
                              p_value <= pvalue_threshold)
            })
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("p_value")
        
        # Cache the result
        cache_key = f"de_{hash(tuple(group_a))}_{hash(tuple(group_b))}"
        self.analysis_cache[cache_key] = result_df
        
        return result_df
    
    def pathway_enrichment(self, gene_list: List[str],
                          background_list: Optional[List[str]] = None,
                          database: str = "KEGG") -> pd.DataFrame:
        """Perform pathway enrichment analysis"""
        
        # This is a simplified version
        # In production, you would connect to actual pathway databases
        
        enrichment_results = []
        
        # Mock pathway data
        pathway_db = {
            "KEGG": {
                "hsa04110": {"name": "Cell cycle", "genes": ["CDK1", "CDK2", "CCNA1", "CCNB1"]},
                "hsa05200": {"name": "Pathways in cancer", "genes": ["TP53", "KRAS", "EGFR", "BRAF"]},
                "hsa04010": {"name": "MAPK signaling pathway", "genes": ["MAPK1", "MAPK3", "RAF1", "EGFR"]}
            },
            "GO": {
                "GO:0006915": {"name": "Apoptotic process", "genes": ["CASP3", "CASP8", "BAX", "BCL2"]},
                "GO:0007049": {"name": "Cell cycle", "genes": ["CDK1", "CDK2", "CCNA1", "CCNB1"]}
            }
        }
        
        if database not in pathway_db:
            raise ValueError(f"Unsupported database: {database}")
        
        pathways = pathway_db[database]
        bg_genes = background_list or gene_list  # Simplified
        
        for pathway_id, pathway_info in pathways.items():
            pathway_genes = set(pathway_info["genes"])
            query_genes = set(gene_list)
            bg_genes_set = set(bg_genes)
            
            # Calculate enrichment
            n_query = len(query_genes)
            n_bg = len(bg_genes_set)
            n_pathway_bg = len(pathway_genes & bg_genes_set)
            n_pathway_query = len(pathway_genes & query_genes)
            
            if n_pathway_bg == 0 or n_query == 0:
                continue
            
            # Fisher's exact test (simplified)
            # In production, use scipy.stats.fisher_exact
            enrichment_ratio = (n_pathway_query / n_query) / (n_pathway_bg / n_bg)
            
            enrichment_results.append({
                "pathway_id": pathway_id,
                "pathway_name": pathway_info["name"],
                "genes_in_pathway": list(pathway_genes & query_genes),
                "n_genes_in_pathway": n_pathway_query,
                "enrichment_ratio": enrichment_ratio,
                "p_value": 0.001 if enrichment_ratio > 2 else 0.5  # Mock p-value
            })
        
        return pd.DataFrame(enrichment_results).sort_values("enrichment_ratio", ascending=False)
    
    def create_interactive_plot(self, data: pd.DataFrame, 
                              plot_type: str = "volcano",
                              **kwargs) -> go.Figure:
        """Create interactive plot for visualization"""
        
        if plot_type == "volcano":
            # Volcano plot for differential expression
            fig = go.Figure()
            
            # Add scatter points
            fig.add_trace(go.Scatter(
                x=data["log2_fold_change"],
                y=-np.log10(data["p_value"]),
                mode='markers',
                marker=dict(
                    size=8,
                    color=data["significant"].map({True: 'red', False: 'blue'}),
                    opacity=0.6
                ),
                text=data["gene"],
                hoverinfo="text+x+y"
            ))
            
            # Add significance thresholds
            log2fc_thresh = kwargs.get("log2fc_threshold", 1.0)
            pval_thresh = kwargs.get("pvalue_threshold", 0.05)
            
            fig.update_layout(
                title="Volcano Plot - Differential Expression",
                xaxis_title="Log2 Fold Change",
                yaxis_title="-Log10(p-value)",
                shapes=[
                    # Vertical lines for fold change thresholds
                    dict(type="line", x0=-log2fc_thresh, x1=-log2fc_thresh,
                         y0=0, y1=1, yref="paper", 
                         line=dict(color="gray", dash="dash")),
                    dict(type="line", x0=log2fc_thresh, x1=log2fc_thresh,
                         y0=0, y1=1, yref="paper",
                         line=dict(color="gray", dash="dash")),
                    # Horizontal line for p-value threshold
                    dict(type="line", x0=0, x1=1, xref="paper",
                         y0=-np.log10(pval_thresh), y1=-np.log10(pval_thresh),
                         line=dict(color="gray", dash="dash"))
                ]
            )
            
        elif plot_type == "pca":
            # PCA plot
            if self.expression_data.empty:
                raise ValueError("No expression data for PCA")
            
            # Prepare data
            X = self.expression_data.T.values
            X = StandardScaler().fit_transform(X)
            
            # Perform PCA
            pca = PCA(n_components=3)
            components = pca.fit_transform(X)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter3d(
                x=components[:, 0],
                y=components[:, 1],
                z=components[:, 2],
                mode='markers',
                marker=dict(size=5),
                text=self.expression_data.columns,
                hoverinfo="text+x+y+z"
            ))
            
            fig.update_layout(
                title="PCA Plot - Gene Expression",
                scene=dict(
                    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                    zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"
                )
            )
            
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        return fig
    
    def save_analysis(self, filepath: str, 
                     format: str = "json") -> bool:
        """Save analysis results to file"""
        try:
            output_data = {
                "metadata": self.metadata,
                "sequences": {k: asdict(v) for k, v in self.sequences.items()},
                "variants": [asdict(v) for v in self.variants],
                "expression_data_shape": self.expression_data.shape,
                "analysis_cache_keys": list(self.analysis_cache.keys()),
                "timestamp": datetime.now().isoformat()
            }
            
            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
            elif format == "pickle":
                with open(filepath, 'wb') as f:
                    pickle.dump(output_data, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Analysis saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            return False
    
    def load_analysis(self, filepath: str,
                     format: str = "json") -> bool:
        """Load analysis results from file"""
        try:
            if format == "json":
                with open(filepath, 'r') as f:
                    data = json.load(f)
            elif format == "pickle":
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Restore data
            self.metadata = data.get("metadata", {})
            
            # Restore sequences
            self.sequences = {}
            for k, v in data.get("sequences", {}).items():
                self.sequences[k] = SequenceData(**v)
            
            # Restore variants
            self.variants = []
            for v in data.get("variants", []):
                self.variants.append(VariantData(**v))
            
            logger.info(f"Analysis loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading analysis: {e}")
            return False

# Example usage and factory function
def create_bioinformatics_pipeline(config_file: str = "bioinformatics_config.json") -> BioinformaticsPipeline:
    """Factory function to create and configure bioinformatics pipeline"""
    pipeline = BioinformaticsPipeline()
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Configure NCBI access
        if "ncbi" in config:
            pipeline.configure_ncbi(
                email=config["ncbi"]["email"],
                api_key=config["ncbi"].get("api_key")
            )
        
        # Load any pre-configured data
        if "data_files" in config:
            for data_type, file_info in config["data_files"].items():
                if data_type == "expression" and file_info.get("path"):
                    pipeline.load_expression_matrix(
                        file_info["path"],
                        format=file_info.get("format", "csv")
                    )
                elif data_type == "variants" and file_info.get("path"):
                    pipeline.load_vcf(file_info["path"])
        
        logger.info("Bioinformatics pipeline initialized successfully")
        
    except FileNotFoundError:
        logger.warning(f"Config file {config_file} not found, using defaults")
    
    return pipeline

# Example usage
async def example_analysis():
    """Example bioinformatics analysis workflow"""
    
    # Create pipeline
    pipeline = create_bioinformatics_pipeline()
    
    # Configure NCBI
    pipeline.configure_ncbi(email="your.email@example.com")
    
    # Fetch sequences
    sequences = await asyncio.gather(
        pipeline.fetch_sequence("NM_001304430", DataSource.NCBI),  # Human TP53
        pipeline.fetch_sequence("P04637", DataSource.UNIPROT),      # TP53 protein
    )
    
    for seq in sequences:
        if seq:
            print(f"Sequence {seq.sequence_id}: {seq.description[:50]}...")
            print(f"  Length: {len(seq.sequence)}, GC%: {seq.gc_content():.1f}")
    
    # Example with expression data
    # Generate mock expression data
    genes = [f"GENE_{i}" for i in range(100)]
    samples_a = [f"CONTROL_{i}" for i in range(10)]
    samples_b = [f"TREATED_{i}" for i in range(10)]
    
    # Create mock DataFrame
    np.random.seed(42)
    data = {}
    for gene in genes:
        # Different means for two groups
        control_mean = np.random.uniform(5, 15)
        treated_mean = control_mean * np.random.uniform(0.5, 2.0)
        
        control_vals = np.random.normal(control_mean, 2, len(samples_a))
        treated_vals = np.random.normal(treated_mean, 2, len(samples_b))
        
        data[gene] = np.concatenate([control_vals, treated_vals])
    
    df = pd.DataFrame(data, index=samples_a + samples_b).T
    pipeline.expression_data = df
    
    # Differential expression analysis
    de_results = pipeline.differential_expression(
        group_a=samples_a,
        group_b=samples_b,
        method="t-test",
        pvalue_threshold=0.01,
        log2fc_threshold=0.5
    )
    
    print(f"\nDifferential Expression Results:")
    print(f"Total genes: {len(de_results)}")
    print(f"Significant genes: {de_results['significant'].sum()}")
    
    # Create volcano plot
    fig = pipeline.create_interactive_plot(
        de_results,
        plot_type="volcano",
        log2fc_threshold=0.5,
        pvalue_threshold=0.01
    )
    
    # Save figure
    fig.write_html("volcano_plot.html")
    print("Volcano plot saved to volcano_plot.html")
    
    # Save analysis
    pipeline.save_analysis("analysis_results.json")

if __name__ == "__main__":
    asyncio.run(example_analysis())