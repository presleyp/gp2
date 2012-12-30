import sys, one_run

num = float(sys.argv[1])

if num <= 20:
    stem_val = 0.8
    tier_val = 0.8
elif num <= 40:
    stem_val = 0.8
    tier_val = 0.4
elif num <= 60:
    stem_val = 0.8
    tier_val = 0
elif num <= 80:
    stem_val = 0.4
    tier_val = 0.8
elif num <= 100:
    stem_val = 0.4
    tier_val = 0.4
elif num <= 120:
    stem_val = 0.4
    tier_val = 0
elif num <= 140:
    stem_val = 0
    tier_val = 0.8
elif num <= 160:
    stem_val = 0
    tier_val = 0.4
else: # <= 180
    stem_val = 0
    tier_val = 0

one_run.run_once(stem_val, tier_val, 0, run_id = 'cluster1-' + str(num))
