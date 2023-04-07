# Assignment 4 - Beam Search

This is an exercise in developing a beam search algorithm for a toy encoder-decoder model.

In this assignment, you will implement some important components of the beam search algorithm.

## Assignment Details

### Important Notes
* Follow instruction in `beam_search.py` to implement `topK()`, `select_hiddens_by_beams()`, and `extract_sequences()` functions.
* We will run your code with the following commands, so make sure that you pass all sanity tests:
```
python test_beam_search.py
```

### Submission
The submission file should be a zip file with the following structure (assuming the campus id is ``CAMPUSID``):
```
CAMPUSID/
├── beam_search.py
├── test_beam_search.py
├── sanity_check.pt
```

### Grading
* 100: You implement all the missing pieces and pass all tests.
* 95: You implement `topK()`, `select_hiddens_by_beams()` functions, and pass the tests.
* 90: You implement `topK()` functions, and pass the tests.
* 85: All missing pieces are implemented, but do not pass all the tests.
* 80: Some parts of the missing pieces are not implemented.

