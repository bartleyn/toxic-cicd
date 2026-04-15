# toxic-cicd


This is an ongoing project I'm taking up in my free time to practice training, updating, and deploying a toxicity detection model (keeping it simple at first) to an API for active use. 

Currently hosted on fly.io, an example usage is:

```
curl -X POST https://toxic-cicd.fly.dev/score \
-H "Content-Type: application/json" \
-d '{"texts":["go screw yourself"], "thresholds":[0.50]}' | jq
```

```
{
  "model_version": "1.1.0",
  "threshold": 0.5,
  "results": [
    {
      "label": 1,
      "scores": {
        "toxicity": 0.7316496663528731,
        "sentiment": -0.10270000249147415,
        "hatespeech": 0.013120003556461961
      }
    }
  ]
}
```
