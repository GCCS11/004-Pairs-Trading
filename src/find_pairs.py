"""
Quick script to test multiple pairs for cointegration.
"""

from data_loader import DataLoader
from pair_selection import PairSelector

# Candidate pairs - same sector/industry
pairs_to_test = [
    ['KO', 'PEP'],  # Beverages
    ['XOM', 'CVX'],  # Oil majors
    ['JPM', 'BAC'],  # Big banks
    ['WMT', 'TGT'],  # Retail
    ['PG', 'CL'],  # Consumer staples
    ['VZ', 'T'],  # Telecom
    ['JNJ', 'PFE'],  # Pharma
    ['BA', 'LMT'],  # Aerospace/Defense
    ['GS', 'MS'],  # Investment banks
    ['HD', 'LOW'],  # Home improvement
]

results = []

for pair in pairs_to_test:
    print(f"\n{'=' * 70}")
    print(f"Testing: {pair[0]} vs {pair[1]}")
    print('=' * 70)

    try:
        # Download data
        loader = DataLoader(tickers=pair)
        data = loader.download_data()

        # Test cointegration
        selector = PairSelector(data)
        corr = selector.calculate_correlation()
        eg = selector.engle_granger_test()
        joh = selector.johansen_test()

        results.append({
            'Pair': f"{pair[0]}-{pair[1]}",
            'Correlation': corr,
            'EG_pvalue': eg['adf_pvalue'],
            'EG_Cointegrated': eg['is_cointegrated'],
            'Johansen_nCoint': joh['n_coint']
        })

    except Exception as e:
        print(f"Error with {pair}: {e}")
        continue

print("\n" + "=" * 70)
print("SUMMARY OF ALL PAIRS")
print("=" * 70)
for r in results:
    coint_status = "YES" if r['EG_Cointegrated'] or r['Johansen_nCoint'] > 0 else "NO"
    print(
        f"{r['Pair']:12} | Corr: {r['Correlation']:.3f} | EG p-val: {r['EG_pvalue']:.4f} | Johansen: {r['Johansen_nCoint']} | Cointegrated: {coint_status}")