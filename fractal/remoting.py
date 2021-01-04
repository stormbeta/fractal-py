# TODO: Experimental

"""
Connect to cloud system like AWS, spin up requested system, run render, then shut down system
1. AWS SDK tooling - should be straightforward
2. Find a suitable AMI (?) - would really really rather not have to bake one
   Alternatively, just leave the instance around, and start/stop it
3. Copy code to machine (python ssh/scp?)
4. Copy resulting render back (only copy merged data!)
   This could be an issue with larger resolutions - will definitely want to zip contents first
   100MP = 400MB in raw data
"""