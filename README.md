# loc2vec

Pytoch implementation of the *Loc2Vec* model outlined in the blog post '*Loc2Vec: Learning Location Embeddings with Triplet-loss Networks*'.

### Use
Implementation requires user to hold both anchor and positive anchor rasters. Negative anchors can be specified, with random indicies selected for such purpose if not.

### Model Results
This implementation was tested with the following OSM raster aggregated channels:
- OSM Lines:
    - roads:
        - ['motorway', 'primary & trunk', 'secondary', 'minor street']
        - ['others']
    -  rails
        - ['others', 'rail']
- OSM Multipolygons
    - ['national park', 'forest', 'grass & park', 'meadow', 'farmyard', 'farmland', 'orchard']
    - ['water']
    - ['industrial', 'construction', 'quarry', 'military']
    - ['railway']
    - ['residential']
    - ['commercial']
    - ['retail']
    - ['allotments', 'cemetary', 'brown field']
    - ['buildings']

