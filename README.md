# Language of Policing

This repository exists for collaborative development of code associated with the NIH-funded "Primed to (re)act" project's computational tasks and analysis.

## Access/authorship

Since it is a private repository, sub-projects/tasks and involved personnel will be recorded here to denote division of labor and facilitate auditing. Access will be granted on a "need to function" basis, thus some contributors' code will be updated by those with access but authorship of that code will always be clear.

Sub-project/task | Personnel
---------------- | ---------
Preprocessing | Chris Graziul (graziul@uchicago.edu)
ASR | Chris Graziul (graziul@uchicago.edu)
VAD | Chris Graziul (graziul@uchicago.edu), Al Hu (huthealex@uchicago.edu)
SER | Ayah Ahmad (ayahahmad@berkeley.edu)
NLP (Penn State) | Pranav Venkit (pranav.venkit@psu.edu)

## Notebooks/documentation

The `docs` and `notebooks` folders have been created to facilitate documentation and development, respectively. For now, feel free to treat these as scratch spaces, though be sure to define variables like working directories so they can run on Midway2/3 or the CDAC machine depending on the task. 

## Pipeline(s)

Data analysis requires several steps of data cleaning, feature extraction, etc. and so it is important to recognize dependencies. Below will be transformed into a graph in time:

```
Preprocessing -> VAD
VAD -> ASR
VAD -> SER
ASR -> NLP
```

Note: NLP is being performed on manually transcribed audio recordings but a critical goal is to rely on ASR to scale NLP to the full corpus.
