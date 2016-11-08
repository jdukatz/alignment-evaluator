ep-ensv-alignref.v2015-10-12
by Lars Ahrenberg

This version is an update from the first version (v2009-12-+08) with the following changes:

Sentence pair 783 in the dev file has been re-aligned.
All files have been checked for omissions and non-existing aligned tokens
All files have been converted to UTF-8



ep-ensv-alignref.v2009-12-08
by Maria Holmqvist

English-Swedish word alignment reference. Sentences from Europarl v.2.

Dev set
-------
Reference word alignment of 972 sentences. Alignment contains
ordinary word links and null links.

Test set
--------
Reference word alignment of 192 sentences. Alignment contains sure,
possible and null links. Annotation was done independently by
two people and the resulting alignments were combined into
one.

Sure: 3340 (0.6357)
Poss: 1237 (0.2354)
Null: 676  (0.1287)
# sentences: 192
# links: 5254
links per sent: 27.36

NAACL format
------------
An Evaluation Exercise for Word Alignment. Rada Mihalcea and Ted Pedersen.
Proceedings of the HLT-NAACL 2004 Workshop on "Building and Using Parallel 
Texts: Data Driven Machine Translation and Beyond", Edmonton, Canada, May 2003. 
http://www.cse.unt.edu/~rada/wpt/papers/pdf/Mihalcea.pdf

Tools
-----
I*Link - fast interactive alignment tool
http://www.ida.liu.se/~nlplab/ILink/ 

Alpaco alignment editor - for adding possible word links and discontinous alignments
http://www-lium.univ-lemans.fr/~lambert/data/epps-alignref.html (Patrik Lambert)
(Original Alpaco editor (Ted Pedersen):  http://www.d.umn.edu/~tpederse/parallel.html)

Lingua::AlignmentSet - toolkit for alignment evaluation and processing e.g., 
converting between formats (BLINKER, TALP, NAACL)
http://gps-tsc.upc.es/veu/personal/lambert/software/AlignmentSet.html
