
API
=====


## List idtracker.ai fragments
```
list-fragments ----session-folder PATH/TO/IDTRACKERAI_RESULTS/session_XXXX
```
where `PATH/TO/IDTRACKERAI_RESULTS/` contains session_* folders and `session_XXXX` is one such folder

## Validate idtracker.ai fragments

```
validate-fragments --experiment-folder PATH/TO/IDTRACKERAI_RESULTS/  --ncores 1
```
where `PATH/TO/IDTRACKERAI_RESULTS/` contains session_* folders

Moreover, the folder  `PATH/TO/IDTRACKERAI_RESULTS/` should contain a file called `corrections.csv`, with the following structure

*corrections.csv*

```
folder,start_end,idtrackerai,human
session_XXXX,148;156,None,2;4
session_XXXX,156;1212,0,2
session_XXXX,156;172,2,4
session_XXXX,1212;1213,None,5;2
```

where

* `folder` refers to the idtrackerai results folder to be corrected

* `start_end` states the first and last frame of the fragment to be corrected, split by `;`

* `idtrackerai` should contain the id (or ids split by `;` for a crossing fragment) that idtracker.ai gave to the fragment

* `human` should contain the id (or ids split by `;` for a crossing fragment) that you as human validator want to provide
