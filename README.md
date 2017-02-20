Face-Detect
===========

Overview
--------
The main idea of this project was to create a form of access control using computer vision techniques. In particular
the Haar-Cascades were used to detect faces and for the facial identification the Local Binary Pattern Histograms
(LBPH) method was used.

Once a face has been detected, it will be checked against the list of authorised users. If there is a match then the
servo motor would be set to open. After a defined timeout, the servo motor will return to its original position.
