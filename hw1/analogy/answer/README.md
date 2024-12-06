# To obtain the wordvector file:
1. python3 analogy/answer/retrofit.py
2. python3 analogy/zipout.py


# To obtain the wordvector magnitude file:
1. python3 analogy/answer/retrofit.py
2. python3 analogy/zipout.py
3. pip install pymagnitude
4. python -m pymagnitude.converter -i retrofitted_vectors_fromwiki.txt -o retrofitted_vectors_fromwiki.magnitude


# To obtain performance metric on the dev set:
1. python analogy/check.py


## The word vector file is too large (350 MB) to submit and push to github or CourSys, so it has to be created by running retrofit.py
