import argparse
from collections import namedtuple
from os import path
from functools import reduce

from pandas.core.window.ewm import zsqrt
import polars as pl
import numpy as np
from numpy.dtypes import StrDType

import gff3_parser
import logging
import pdb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

vcf_field = namedtuple("VCFField", ["chr", "position", "ref", "alt", "transcript_id", "vep_csq", "gene_id", "gene_symbol", "feature_type", "biotype", "loftee", "rescue", "rescue_prob", "rescue_type"])
sample_field = np.dtype([('chr', StrDType), ('position', np.uint64), ('ref', StrDType), ('alt', StrDType), ('tid_idx', np.uint32), ('exists', np.bool_), ('rescue', np.bool_), ('rescue_prob', np.float32)])
carrier_field = np.dtype([('carrier', np.bool_), ('rescue', np.bool_), ('rescue_prob', np.float32)])

def parse_args():
    parser = argparse.ArgumentParser(description="Analyse 1000Genomes RescueRanger and RSEM extractions.")
    parser.add_argument("--gff3", help="GFF3 file containing all possible transcripts.")
    parser.add_argument("--analysis_pairs_file", help="File containing tab separated pairs of RescueRanger and RSEM extraction files.")

    return parser.parse_args()


def parse_tid_gene_map_from_gff3(file: str):
    gff3 = gff3_parser.parse_gff3(file, parse_attributes=True, verbose=False)
    gff3_transcripts = gff3[gff3.Type == 'mRNA']

    return pl.from_pandas(gff3_transcripts).select(['transcript_id', 'Parent', 'Name']).with_columns(
        pl.col('Parent').str.split(':').list.get(1).alias('gene_id'), 
        pl.col('Name').str.split('-').list.get(0).alias('symbol')
    ).select(['transcript_id', 'gene_id', 'symbol'])


def read_rsem_file(file: str) -> pl.DataFrame:
    return pl.read_csv(file, separator="\t") \
        .select(["transcript_id", "TPM"]) \
        .rename({ "TPM": path.basename(file).split(".")[0] })


def read_vcf_file(file: str, transcript_index_map: pl.DataFrame) -> pl.DataFrame:
    df = pl.read_csv(file, separator="\t") \
            .filter(
                (pl.col("INFO/RESCUE") != ".") & 
                (pl.col("INFO/RESCUE_PROB") != ".") &
                (pl.col("INFO/RESCUE_TYPE") != ".") 
            ).with_columns(
                pl.col("INFO/vepConsequence").str.split(","),
                pl.col("INFO/vepGene").str.split(","),
                pl.col("INFO/vepSYMBOL").str.split(","),
                pl.col("INFO/vepFeature_type").str.split(","),
                pl.col("INFO/vepFeature").str.split(","),
                pl.col("INFO/vepBIOTYPE").str.split(","),
                pl.col("INFO/vepLoF").str.split(","),
                pl.col("INFO/RESCUE").str.split(","),
                pl.col("INFO/RESCUE_PROB").str.split(","),
                pl.col("INFO/RESCUE_TYPE").str.split(",")
            ).explode([
                'INFO/vepConsequence', 'INFO/vepGene', 'INFO/vepSYMBOL',
                'INFO/vepFeature_type', 'INFO/vepFeature', 'INFO/vepBIOTYPE',
                'INFO/vepLoF', 'INFO/RESCUE', 'INFO/RESCUE_PROB', 'INFO/RESCUE_TYPE'
            ]).filter(
                (pl.col("INFO/vepFeature_type") == "Transcript") & 
                (pl.col("INFO/vepLoF") == "HC") & 
                (pl.col("INFO/vepBIOTYPE") == "protein_coding") &
                ((pl.col("INFO/RESCUE_PROB").str.contains(r"^\.")) | (pl.col("INFO/RESCUE_PROB").str.contains(r"\.$"))) &
                (pl.col("INFO/vepConsequence").is_in(["stop_gained", "frameshift_variant", "splice_donor_variant", "splice_acceptor_variant"]))
            ).with_columns(
                pl.col("INFO/RESCUE_PROB").str.split("&"),
                pl.col("INFO/RESCUE").str.split("&")
            ).explode(["INFO/RESCUE_PROB", "INFO/RESCUE"]).filter((pl.col("INFO/RESCUE_PROB") != ".") & (pl.col("INFO/RESCUE") != ".")) \
            .select(["CHROM", "POS", "REF", "ALT", "INFO/vepFeature", "INFO/RESCUE", "INFO/RESCUE_PROB"]) \
            .rename({
                "CHROM": "chr",
                "POS": "pos",
                "REF": "ref",
                "ALT": "alt",
                "INFO/vepFeature": "transcript_id",
                "INFO/RESCUE": "rescue",
                "INFO/RESCUE_PROB": "rescue_prob"
            }) \
            .join(transcript_index_map, on="transcript_id", how="left") \
            .drop("transcript_id")

    return df


def z_score(carrier_row, tpm_row):
    control_mask = ~(carrier_row['carrier'])
    controls = tpm_row[control_mask]

    if len(controls) == 0:
        return np.NaN

    mean = np.mean(controls)
    std = np.std(controls)

    if std == 0.0:
        return np.NaN

    return (tpm_row - mean) / std


def main():
    args = parse_args()

    # Process GFF file and get all transcript to gene mappings
    logging.info(f"Reading in GFF3 file: {args.gff3}...")
    transcript_gene_map = parse_tid_gene_map_from_gff3(args.gff3)
    transcript_gene_map = transcript_gene_map.with_columns(pl.arange(0, transcript_gene_map.height).alias("transcript_idx"))

    # Read in file pairs
    df_file_pairs = pl.read_csv(args.analysis_pairs_file, separator="\t")
    logging.info(f"Found {df_file_pairs.shape[0]} VCF - RSEM file pairs.")

    # Creates a dataframe with TPM values for transcripts per sample: (sample1, ..., sample_n, transcript_idx)
    logging.info(f"Reading and joining RSEM files...")
    rsems = [read_rsem_file(rsem_file) for rsem_file in df_file_pairs["rsems"].to_list()]
    df_rsem = reduce(lambda df1, df2: df1.join(df2, on="transcript_id", how="full").drop("transcript_id_right"), rsems)
    df_rsem = df_rsem.join(transcript_gene_map.select(["transcript_id", "transcript_idx"]), on="transcript_id", how="left").drop("transcript_id")

    # Filter to median TPM > 1
    df_rsem = df_rsem.drop_nulls("transcript_idx") \
        .with_columns(pl.concat_list(df_rsem.columns[:-1]).list.median().alias("row_median")) \
        .filter(pl.col("row_median") > 1.0) \
        .select(pl.exclude("row_median"))

    # Convert dataframe to python dict for quick access of TPM values per transcript_idx value
    dict_rsem = {int(row[-1]): row[:-1] for row in df_rsem.to_numpy() }

    logging.info(f"Reading and joining VCF files...")
    vcfs = [read_vcf_file(vcf_file, transcript_gene_map.select(["transcript_id", "transcript_idx"])) for vcf_file in df_file_pairs["vcfs"].to_list()]
    variant_transcripts_per_vcf = [df.select(["chr", "pos", "ref", "alt", "transcript_idx"]) for df in vcfs]
    df_variant_transcripts = reduce(lambda df1, df2: df1.vstack(df2), variant_transcripts_per_vcf).unique(keep='any')

    # Is the mapping not just extra work? Probably not as later work (which is exponential) should be easier. Does this remain true when hashing?  TODO: Think about this when not tired
    # Get a integer mapping for chromosomes, so we don't have to do as many string comparisons later on
    chr_idx_map = df_variant_transcripts.select(["chr"]).unique()
    chr_idx_map = chr_idx_map.with_columns(pl.arange(0, chr_idx_map.height).alias("chr_idx"))

    # Also create integer mappings for ref and alt!
    ref_idx_map = df_variant_transcripts.select(["ref"]).unique()
    ref_idx_map = ref_idx_map.with_columns(pl.arange(0, ref_idx_map.height).alias("ref_idx"))
    alt_idx_map = df_variant_transcripts.select(["alt"]).unique()
    alt_idx_map = alt_idx_map.with_columns(pl.arange(0, alt_idx_map.height).alias("alt_idx"))

    # Replace chr, ref, alt by the integer mappings
    df_variant_transcripts = df_variant_transcripts \
        .join(chr_idx_map, on="chr", how="left").drop("chr") \
        .join(ref_idx_map, on="ref", how="left").drop("ref") \
        .join(alt_idx_map, on="alt", how="left").drop("alt") \
        .select(["chr_idx", "pos", "ref_idx", "alt_idx", "transcript_idx"])

    # Also replace chr, ref, alt for individual dataframes
    # Should add a cast for rescue and rescue_prob just to make sure there's no need to do so later on?
    vcfs = [
        df \
            .join(chr_idx_map, on="chr", how="left").drop("chr") \
            .join(ref_idx_map, on="ref", how="left").drop("ref") \
            .join(alt_idx_map, on="alt", how="left").drop("alt") \
            .select(["chr_idx", "pos", "ref_idx", "alt_idx", "transcript_idx", "rescue", "rescue_prob"])
        for df in vcfs
    ]

    # Convert individual dataframes to dicts for fast lookup
    vcfs = [ { row[0:5]: row[5:] for row in df.rows() } for df in vcfs ]

    # Initialize empty matrix of (transcript,variant) by sample fields
    carrier_rescues = np.zeros((df_variant_transcripts.height, len(vcfs)), dtype=carrier_field)
    for sample_idx, dictionary in enumerate(vcfs): # Hashing is still king for lookups, so dicts are the way to go
        for row_idx, row in enumerate(df_variant_transcripts.rows()):
            key = tuple(row[0:5])
            if key not in dictionary:
                continue

            carrier_rescues[row_idx, sample_idx] = (True, *dictionary[key])

    # Numpy arrays for easy access
    variant_transcripts = df_variant_transcripts.to_numpy()

    # Calculate z-scores per sample
    logging.info("Calculating Z scores...")
    z_scores = np.zeros((df_variant_transcripts.height, len(vcfs)), dtype=np.float32)
    used_rsem = np.zeros((df_variant_transcripts.height, len(vcfs)), dtype=np.float32)
    # pdb.set_trace()
    n_within = 0
    for idx, carrier_row in enumerate(carrier_rescues):
        tidx = variant_transcripts[idx, 4]
        if tidx not in dict_rsem:
            z_scores[idx, :] = np.NaN
            continue

        n_within += 1
        rsem_row = dict_rsem[tidx]
        z_scores[idx, :] = z_score(carrier_row, rsem_row)
        used_rsem[idx, :] = rsem_row
    logging.info(f"Calculated z_score for {n_within}/{carrier_rescues.shape[0]} rows.")


    logging.info("Preparing output file...")
    dfs = [
        df_variant_transcripts,
        pl.DataFrame({"sample": np.array([df_rsem.columns[:-1]] * df_variant_transcripts.height), "tpm": used_rsem }),
        pl.DataFrame({"n_carriers": np.sum(carrier_rescues['carrier'], axis=1), "n_rescues": np.sum(carrier_rescues['rescue'], axis=1) }),
        pl.DataFrame(carrier_rescues),
        pl.DataFrame({"z_score": z_scores})
    ]

    df_out = pl \
        .concat(dfs, how="horizontal") \
        .join(chr_idx_map, on="chr_idx", how="left").drop("chr_idx") \
        .join(ref_idx_map, on="ref_idx", how="left").drop("ref_idx") \
        .join(alt_idx_map, on="alt_idx", how="left").drop("alt_idx") \
        .join(transcript_gene_map, on="transcript_idx", how="left").drop("transcript_idx") \
        .explode(["sample", "tpm", "carrier", "rescue", "rescue_prob", "z_score"]) \
        .filter(pl.col("z_score").is_not_nan()) \
        .select(["chr", "pos", "ref", "alt", "transcript_id", "gene_id", "symbol", "n_carriers", "n_rescues", "sample", "tpm", "carrier", "rescue", "rescue_prob", "z_score"]) \
        .sort(["chr", "pos", "ref", "alt", "transcript_id"])

    logging.info("Writing to: tpm_table.tsv")
    print(df_out)
    df_out.write_csv("tpm_table.tsv", separator="\t")


if __name__ == "__main__":
    main()

