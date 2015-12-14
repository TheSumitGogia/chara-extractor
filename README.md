chara-extractor
===============

Overview
--------

__chara-extractor__ contains a set of tools for building classifiers to extract salient characters and 
identify salient relationships from novels. At the time of writing, not all planned functions, such
as accurate relationship identification and semantic relationship labeling, are provided. 

However, users
can still train character extractors using our tools, as well as relatively inaccurate relationship identifiers.
It is particularly easy to access and test the character extractors and relationship identifiers trained by us
and discussed in our paper (), allowing for verification of the results presented there. 

It should be noted
that there are a large number of details in implementation that are not highlighted in the paper - these
include strategies for sectioning various novels into sentences, paragraphs and chapters, details for how we
manage disambiguation in the evaluation stage when a name appears to refer to multiple characters, and many
more. While it is not necessary to examine these to use our pre-trained classifiers, for better understanding
of how to reproduce our results from scratch it would be good to look at them in our lower level modules. 

Setup
-----

Our toolset requires a number of PyPI packages to run correctly. These are listed in the requirements
file in the top-level directory. We are unaware if automatic installation using this file and the
package manager Pip is doable - but running pip install with each of the packages listed should work.
In addition to the packages listed here, some extra work will need to be done to get all our code
running:

1. You must have a build of Stanford's CoreNLP toolset, with the top directory for that project stored
   in the environment variable `CORE_NLP`. If you have no BASH script called `corenlp.sh` in there,
   something is seriously wrong.

2. You must have WordNet installed with the NLTK package; to perform the installation after installing
   NLTK, open up a command line and attempt to access WordNet from NLTK corpus. NLTK will bring up
   a GUI that you can use to install WordNet. 

3. To reproduce our results, and likely for your own organization as well, you will want to download
   our full data folder from MIT Dropbox:

    <https://www.dropbox.com/s/af02rfka01sbtdu/chara_extractor_data.tar.gz?dl=0> 

   and unzip it at the level of the main scripts (`evaluate`, `collect`,...). 
   While our scripts provide decent high-level functionality
   for collecting all the data, it is extremely time-consuming, taking on the order of days for
   most CPUs. In addition, it will be difficult to reproduce the manual checks we had for ensuring
   all novels we found in Project Gutenberg corresponded with those we found in Sparknotes. 

Usage
-----

Our code is accessible at a number of levels, with the highest being simplest to use and obscuring
many parameter details, and the lowest allowing for very fine-tuned access. For example, 
at the highest level candidate extraction acts out with the default behavior we found to be
useful, but in lower level modules you can access the candidate extraction functions and pass
parameters for values such as absolute frequency threshold, limits on numbers of candidates for
different amounts of tokens, or even top percentage of candidates to extract by frequency.

### Classifier Result Reproduction
To test the classifiers we trained, one only has to use the top-level script `evaluate`. Evaluate
provides two subcommands `quant` and `qual` for getting quantitative and qualitative results
respectively. Each has options for specifying trained classifier directory, feature directory, and label
directory, and `qual` has an option for specifying particular books to test on. The command
below (run from the top directory) will get precision/recall for our best classifiers 
on characters and relationships:

    python evaluate.py quant -t char -c data/classifiers/charclf
    python evaluate.py quant -t rel -c data/classifiers/pairclf

The following command will qualitatively test sample books _Crime and Punishment_ and _Secret Garden_,
outputting the characters found by the classifier and those listed in Sparknotes; the characters
our classifier finds that are duplicates according to our disambiguation as well as those
that don't concur with Sparknotes are listed alone as well for clarity. It is as straightforward
to test with different books and different numbers of books as it seems - however, you must
use the shorthand book names from `data/raw`. Note that we have not implemented this for
relationships due to clearly poor performance in training - it makes it difficult to use
qualitative results for information gain. 

    python evaluate.py qual -t char -b crime secretgarden -c data/classifiers/charclf

### Baseline Result Reproduction
You can also reproduce our baseline results. It's as easy as omitting the classifier directory
and adding one argument.

    python evaluate.py quant -t char --baseline
    python evaluate.py quant -t rel --baseline

For qualitative character results:
        
    python evaluate.py qual -t char --baseline -b crime secretgarden 

### Digging Deeper
Besides reproducing our results from our classifiers, one can attempt to reproduce our results
more completely by running every part of the pipeline described in our paper. Data collection,
labeling, and feature extraction are all available through the command line tools
`collect.py`, `label.py`, and `features.py`. All of these top level scripts give access
to some parameters which we pre-determine for the paper; the most control will of course
however be found by digging into our lower-level scripts in the `chara` directory.
