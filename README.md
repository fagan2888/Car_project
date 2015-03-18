Car Preference Clustering
=========================

This is my project for Zipfian Academy. Basically, I create a recommendation
system for car rentals based on where people are coming from and where they
are going.

This repo contains the following files:

1. country_clustering.py --- This contains all the code I used for analysis. It also has it's own readme/instructions commented in the code.
2. world_cc_map.html --- This is the html/CSS/javascript/d3 code I used to create the visual representation of my project. You can view this visualization at bl.ocks.org/warrenronsiek.
3. cc_info.json, incoming_info.json, orig_dest.json --- These are results from the country_clustering.py file written into json format to be used by world_cc_map.html.

Note that this repo does not include the original dataset, code I used to
clean the dataset, or code for feature engineering. This is due to an implicit
NDA I have with the providers of the data.
