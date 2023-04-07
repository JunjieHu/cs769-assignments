# Assignment 3 (Course project proposal)

## Logistics
The class project will be carried out in groups of 2 or 3 people. Please actively search your teammate either posting a note on piazza or chatting after class. We encourage everyone to submit your interested topics to the survey link. If you really have difficulty finding a teammate, please let the instructor and TA know by 3/20. The deadline of Assignment 3 is 3/29 11:59PM. Every team has 3 late days in total for Assignment 3 and 5.

## Submission Information
In **Homework Assignment 3**, in addition to report results of a re-implementation of an existing work, you should also turn in a brief report (project proposal) that provides an overview of your idea and also contains a brief survey of related work on the topic. Please submit the following two files to [Canvas](https://canvas.wisc.edu/courses/343092/assignments).


- **a link to a github repository containing your code (assignments 3 and 5)**: This should be a single line file "github.txt" in the top directory. Your github repository must be viewable to the TAs and instructor by the submission deadline. If your repository is private make it accessible to the TAs by the submission deadline **(Github ID: junjiehu, HuiyuBao)**. If your repository is not visible to the TAs, your assignment will not be considered complete, so if you are worried please submit well in advance of the deadline so we can confirm the submission is visible. We use this repository to check contributions of all team members.

- **a report (assignments 3 and 5)**: This should be named "report.pdf" in the top directory. This is for assignments 3 and 5, and can be **up to 7 pages for assignment 3 and 9 pages for assignment 5**. References are not included in the page count, and it is OK to submit appendices that include supplementary information such as hyperparameter settings or additional output examples, although there is no guarantee that the TAs will read them. Submissions that exceed the page count will be penalized one third grade (33%) for each page over. The report must use the official ACL style templates, which are available from [here](https://github.com/acl-org/acl-style-files) (Latex and Word). Please follow the paper formatting guidelines general to *ACL* conferences available [here](https://acl-org.github.io/ACLPUB/formatting.html). The proposal report should include the following information:

    - Project title and list of group members.
    - Overview of project idea. This should be approximately one page long.
    - A short literature survey of 4 or more relevant papers. The literature review should take up approximately one page.
    - Reimplementation of one related work, report your scores on their dataset, and perform fine-grained analysis on the existing method. This should be approximately 1-2 page long.
    - Description of potential methods beyond the existing work. This should be approximately one page long.
    - Description of potential data sets to use for your experiments, the experimental settings, and the experiments to be conducted. This should be approximately one page long.
    - Plan of activities, including what you plan to complete by the final report and how you plan to divide up the work among the group members. This should be approximately one page long.


## Grading
The grading breakdown for the proposal is as follows:

- 25% for clear and concise description of proposed method
- 25% for literature survey that covers at least 4 relevant papers
- 25% for the reimplementation of one existing work.
- 15% for plan of activities
- 10% for quality of writing

## Expectation of Re-implementation
You should pick one recent paper (ideally within the past three years) after conducting the literature survey. In the experiment section, please provide a table containing the original scores reported in your selected paper and the scores of your re-implemented models. You're allowed to use any libraries for the implementation. 
- If you plan to re-implement the selected paper without reusing any existing code base, please follow the detailed instructions in their paper by strictly following their model architecture and hyperparameters. Your reimplementation scores should not be too far away from the scores reported in their paper (e.g., within 2-5 accuracy points, depending on the difficulty of tasks and metrics).
- If you plan to re-use their code base for more ambitious research problems (e.g., new challenges unexplored in the literature), you should **propose a few extensions** on top of their method and perform a detailed comparison with their original model's output in the experimental section. **Simply re-running the existing code base is not sufficient**. The extensions will be regarded as exploring potential ideas and preliminary results for HW5. Please also **clearly specify which parts are implemented by yourself** in the report. 

## Potentail Topics
The project is an integral part of this class, and is designed to be as similar as possible to researching and writing a conference-style paper. We provide a list of suggested project topics below for you to choose from, though you may discuss other project ideas with us.

- [Any SemEval 2021 Task](https://semeval.github.io/SemEval2021/tasks)
- [X-FACTR multilingual knowledge probing in QA](https://x-factr.github.io/)
- [iSarcasm Sarcasm Detection Dataset](https://github.com/silviu-oprea/iSarcasm)
- [GoEmotions Fine-grained Emotion Detection Dataset](https://github.com/google-research/google-research/tree/master/goemotions)
- [SciREX Scientific Information Extraction](https://github.com/allenai/SciREX)
- [Subjective Intent Classification in Discourse](https://github.com/elisaF/subjective_discourse)
- [Very Low Resource MT](http://statmt.org/wmt21/unsup_and_very_low_res.html)
- [National NLP Clinical Challenges (n2c2)](https://n2c2.dbmi.hms.harvard.edu/2022-track-2)
- [MultiWoZ: Task-oriented dialog](https://github.com/budzianowski/multiwoz)
- [XTREME: Zero-shot Cross-lingual Transfer](https://github.com/google-research/xtreme)
- [MIA: Cross-lingual Open-Retrieval QA](https://mia-workshop.github.io/shared_task.html)

*Notes:* We are organizing a multilingual workshop (MIA) at NAACL 2022 which includes a shared task on [cross-lingual open-retrieval QA](https://mia-workshop.github.io/shared_task.html). Please feel free to consider this as your course project and discuss any potential ideas with us.
