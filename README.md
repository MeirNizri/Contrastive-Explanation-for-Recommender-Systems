# Contrastive Explanations for Recommendation Systems

## Abstract
Recommendation systems are widely used and are present in many applications, such as movie recommendation, product sales, and content providers. However, current recommendation systems are usually stern and lack the ability to explain their decisions or allow the users to question them. 

In this paper, we develop an automatic method that, given a contrastive query from the user, generates contrastive explanations based on items' features and users' preferences (provided as ratings). That is, once receiving a recommendation, the users have the option to ask the system why it did not recommend a specific different item. Our method enables a recommendation system to reply with a meaningful and convincing personalized explanation. For example, the recommendation system may recommend the user to buy a Samsung S22 phone. The user may ask the system why it did not recommend the Xiaomi 12. Based on the user's preferences, all other users' preferences, and the specific phones in question, our method might infer that a good camera is very important to the user, and thus, say that the Samsung S22 includes a better camera than the Xiaomi 12.

We compose a new data-set based on user ratings of the most popular cell phones in the US in 2022.
Based on this data-set, we run an experiment with 100 human participants who are recommended an item and shown contrastive explanations generated by our method, as well as two additional baseline-methods.
We show that humans are more convinced that the recommended item is better than the contrastive item when using our contrastive explanations.
