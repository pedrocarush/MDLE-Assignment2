import os
import subprocess

KS = [11]
CDT_STDS = [2]
CS_MVTS = [1.001, 1.01, 1.02, 1.1, 1.5]
DBSCAN_EPSES = [1000,2000,3000,5000,10000]
EXCLUDE_COMPRESSION_SETS = [True, False]

if __name__ == '__main__':

    if os.path.exists('exercise1_tries_progress.txt'):
        with open('exercise1_tries_progress.txt', 'r') as f:
            skip_n = int(f.readline())
    else:
        skip_n = 0

    n = 0

    try:
        for k in KS:
            for cdt_std in CDT_STDS:
                for cs_mvt in CS_MVTS:
                    for dbscan_eps in DBSCAN_EPSES:
                        for exclude_compression_sets in EXCLUDE_COMPRESSION_SETS:
                            if n >= skip_n:
                                subprocess.run(['spark-submit', 'exercise1.py',
                                    '--bfr-n-clusters', str(k),
                                    '--bfr-cdt-std', str(cdt_std),
                                    '--bfr-cs-mvt', str(cs_mvt),
                                    '--bfr-dbscan-eps', str(dbscan_eps),
                                ] + (['--bfr-exclude-compression-sets'] if exclude_compression_sets else []))

                            n += 1
    
    except KeyboardInterrupt:
        with open('exercise1_tries_progress.txt', 'w') as f:
            f.write(str(n))
        exit(0)