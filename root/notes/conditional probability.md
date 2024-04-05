Conditional probability
---

Conditional probability is a concept in probability theory that deals with the likelihood of an event occurring given that another event has already occurred. It expresses the probability of one event happening under the condition that another event has occurred. 

Mathematically, the conditional probability of an event $( A )$ occurring given that an event $( B )$ has already occurred is denoted by $( P(A|B) )$, read as "the probability of $( A )$ given $( B )$" and is calculated using the following formula:

$[ P(A|B) = \frac{P(A \cap B)}{P(B)} ]$

Where:
- $( P(A|B) )$ is the conditional probability of event $( A )$ given event $( B )$.
- $( P(A \cap B) )$ is the probability of both events $( A )$ and $( B )$ occurring simultaneously (i.e., the joint probability of $( A )$ and $( B )$).
- $( P(B) )$ is the probability of event $( B )$ occurring.

This formula essentially states that the probability of $( A )$ given $( B )$ is the ratio of the probability of both $( A )$ and $( B )$ occurring together to the probability of $( B )$ occurring.

Here's a basic example to illustrate conditional probability:

Suppose you have a standard deck of 52 playing cards. You draw one card at random. Let:
- Event $( A )$: Drawing a red card.
- Event $( B )$: Drawing a heart.

The conditional probability of drawing a red card given that the drawn card is a heart ($( P(A|B) )$) can be calculated using the formula mentioned above:

$[ P(A|B) = \frac{P(A \cap B)}{P(B)} ]$

Here, $( P(A \cap B) )$ is the probability of drawing a red heart card, which is $( \frac{1}{52} )$ because there is only one red heart card in the deck.

$( P(B) )$ is the probability of drawing any heart card, which is $( \frac{13}{52} )$ because there are 13 hearts in the deck.

So, $( P(A|B) = \frac{\frac{1}{52}}{\frac{13}{52}} = \frac{1}{13} )$.

This means that given you've drawn a heart, the probability of it being red is $( \frac{1}{13} )$.