[Perplexity home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/perplexity/logo/SonarByPerplexity.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/perplexity/logo/Sonar_Wordmark_Light.svg)](https://docs.perplexity.ai/home.mdx)

Search docs

Ctrl K

Search...

Navigation

Perplexity API

Chat Completions

[Home](https://docs.perplexity.ai/home) [Guides](https://docs.perplexity.ai/guides/getting-started) [API Reference](https://docs.perplexity.ai/api-reference/chat-completions) [Changelog](https://docs.perplexity.ai/changelog/changelog) [System Status](https://docs.perplexity.ai/system-status/system-status) [FAQ](https://docs.perplexity.ai/faq/faq) [Discussions](https://docs.perplexity.ai/discussions/discussions)

POST

/

chat

/

completions

Try it

cURL

Python

JavaScript

PHP

Go

Java

Copy

```
curl --request POST \
  --url https://api.perplexity.ai/chat/completions \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: application/json' \
  --data '{
  "model": "sonar",
  "messages": [\
    {\
      "role": "system",\
      "content": "Be precise and concise."\
    },\
    {\
      "role": "user",\
      "content": "How many stars are there in our galaxy?"\
    }\
  ],
  "max_tokens": 123,
  "temperature": 0.2,
  "top_p": 0.9,
  "search_domain_filter": null,
  "return_images": false,
  "return_related_questions": false,
  "search_recency_filter": "<string>",
  "top_k": 0,
  "stream": false,
  "presence_penalty": 0,
  "frequency_penalty": 1,
  "response_format": null
}'
```

200

400

401

429

500

504

524

Copy

```
{
  "id": "3c90c3cc-0d44-4b50-8888-8dd25736052a",
  "model": "sonar",
  "object": "chat.completion",
  "created": 1724369245,
  "citations": [\
    "https://www.astronomy.com/science/astro-for-kids-how-many-stars-are-there-in-space/",\
    "https://www.esa.int/Science_Exploration/Space_Science/Herschel/How_many_stars_are_there_in_the_Universe",\
    "https://www.space.com/25959-how-many-stars-are-in-the-milky-way.html",\
    "https://www.space.com/26078-how-many-stars-are-there.html",\
    "https://en.wikipedia.org/wiki/Milky_Way"\
  ],
  "choices": [\
    {\
      "index": 0,\
      "finish_reason": "stop",\
      "message": {\
        "role": "assistant",\
        "content": "The number of stars in the Milky Way galaxy is estimated to be between 100 billion and 400 billion stars. The most recent estimates from the Gaia mission suggest that there are approximately 100 to 400 billion stars in the Milky Way, with significant uncertainties remaining due to the difficulty in detecting faint red dwarfs and brown dwarfs."\
      },\
      "delta": {\
        "role": "assistant",\
        "content": ""\
      }\
    }\
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 70,
    "total_tokens": 84
  }
}
```

#### Authorizations

[​](https://docs.perplexity.ai/api-reference/chat-completions#authorization-authorization)

Authorization

string

header

required

Bearer authentication header of the form `Bearer <token>`, where `<token>` is your auth token.

#### Body

application/json

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-model)

model

string

required

The name of the model that will complete your prompt. Refer to [Supported Models](https://docs.perplexity.ai/guides/model-cards) to find all the models offered.

Example:

`"sonar"`

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-messages)

messages

object\[\]

required

A list of messages comprising the conversation so far.

Showchild attributes

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-messages-content)

messages.content

string

required

The contents of the message in this turn of conversation.

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-messages-role)

messages.role

enum<string>

required

The role of the speaker in this turn of conversation. After the (optional) system message, user and assistant roles should alternate with `user` then `assistant`, ending in `user`.

Available options:

`system`,

`user`,

`assistant`

Example:

```json
[\
  {\
    "role": "system",\
    "content": "Be precise and concise."\
  },\
  {\
    "role": "user",\
    "content": "How many stars are there in our galaxy?"\
  }\
]

```

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-max-tokens)

max\_tokens

integer

The maximum number of completion tokens returned by the API. The number of tokens requested in `max_tokens` plus the number of prompt tokens sent in messages must not exceed the context window token limit of model requested. If left unspecified, then the model will generate tokens until either it reaches its stop token or the end of its context window.

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-temperature)

temperature

number

default:0.2

The amount of randomness in the response, valued between 0 inclusive and 2 exclusive. Higher values are more random, and lower values are more deterministic.

Required range: `0 <= x < 2`

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-top-p)

top\_p

number

default:0.9

The nucleus sampling threshold, valued between 0 and 1 inclusive. For each subsequent token, the model considers the results of the tokens with top\_p probability mass. We recommend either altering top\_k or top\_p, but not both.

Required range: `0 <= x <= 1`

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-search-domain-filter)

search\_domain\_filter

any\[\]

Given a list of domains, limit the citations used by the online model to URLs from the specified domains. Currently limited to only 3 domains for whitelisting and blacklisting. For **blacklisting** add a `-` to the beginning of the domain string. **Only available in certain tiers** \- refer to our usage tiers [here](https://docs.perplexity.ai/guides/usage-tiers).

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-return-images)

return\_images

boolean

default:false

Determines whether or not a request to an online model should return images. **Only available in certain tiers** \- refer to our usage tiers [here](https://docs.perplexity.ai/guides/usage-tiers).

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-return-related-questions)

return\_related\_questions

boolean

default:false

Determines whether or not a request to an online model should return related questions. **Only available in certain tiers** \- refer to our usage tiers [here](https://docs.perplexity.ai/guides/usage-tiers).

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-search-recency-filter)

search\_recency\_filter

string

Returns search results within the specified time interval - does not apply to images. Values include `month`, `week`, `day`, `hour`.

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-top-k)

top\_k

number

default:0

The number of tokens to keep for highest top-k filtering, specified as an integer between 0 and 2048 inclusive. If set to 0, top-k filtering is disabled. We recommend either altering top\_k or top\_p, but not both.

Required range: `0 <= x <= 2048`

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-stream)

stream

boolean

default:false

Determines whether or not to incrementally stream the response with [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format) with `content-type: text/event-stream`.

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-presence-penalty)

presence\_penalty

number

default:0

A value between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Incompatible with `frequency_penalty`.

Required range: `-2 <= x <= 2`

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-frequency-penalty)

frequency\_penalty

number

default:1

A multiplicative penalty greater than 0. Values greater than 1.0 penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. A value of 1.0 means no penalty. Incompatible with `presence_penalty`.

Required range: `x > 0`

[​](https://docs.perplexity.ai/api-reference/chat-completions#body-response-format)

response\_format

object

Enable structured outputs with a JSON or Regex schema. Refer to the guide [here](https://docs.perplexity.ai/guides/structured-outputs) for more information on how to use this parameter. **Only available in certain tiers** \- refer to our usage tiers [here](https://docs.perplexity.ai/guides/usage-tiers).

#### Response

200

200400401429500504524

application/json

application/jsontext/event-stream

OK

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-id)

id

string

An ID generated uniquely for each response.

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-model)

model

string

The model used to generate the response.

Example:

`"sonar"`

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-object)

object

string

The object type, which always equals `chat.completion`.

Example:

`"chat.completion"`

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-created)

created

integer

The Unix timestamp (in seconds) of when the completion was created.

Example:

`1724369245`

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-citations)

citations

any\[\]

Citations for the generated answer.

Example:

```json
[\
  "https://www.astronomy.com/science/astro-for-kids-how-many-stars-are-there-in-space/",\
  "https://www.esa.int/Science_Exploration/Space_Science/Herschel/How_many_stars_are_there_in_the_Universe",\
  "https://www.space.com/25959-how-many-stars-are-in-the-milky-way.html",\
  "https://www.space.com/26078-how-many-stars-are-there.html",\
  "https://en.wikipedia.org/wiki/Milky_Way"\
]

```

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-choices)

choices

object\[\]

The list of completion choices the model generated for the input prompt.

Showchild attributes

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-choices-index)

choices.index

integer

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-choices-finish-reason)

choices.finish\_reason

enum<string>

The reason the model stopped generating tokens. Possible values include `stop` if the model hit a natural stopping point, or `length` if the maximum number of tokens specified in the request was reached.

Available options:

`stop`,

`length`

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-choices-message)

choices.message

object

The message generated by the model.

Showchild attributes

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-choices-message-content)

choices.message.content

string

required

The contents of the message in this turn of conversation.

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-choices-message-role)

choices.message.role

enum<string>

required

The role of the speaker in this turn of conversation. After the (optional) system message, user and assistant roles should alternate with `user` then `assistant`, ending in `user`.

Available options:

`system`,

`user`,

`assistant`

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-choices-delta)

choices.delta

object

The incrementally streamed next tokens. Only meaningful when `stream = true`.

Showchild attributes

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-choices-delta-content)

choices.delta.content

string

required

The contents of the message in this turn of conversation.

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-choices-delta-role)

choices.delta.role

enum<string>

required

The role of the speaker in this turn of conversation. After the (optional) system message, user and assistant roles should alternate with `user` then `assistant`, ending in `user`.

Available options:

`system`,

`user`,

`assistant`

Example:

```json
[\
  {\
    "index": 0,\
    "finish_reason": "stop",\
    "message": {\
      "role": "assistant",\
      "content": "The number of stars in the Milky Way galaxy is estimated to be between 100 billion and 400 billion stars. The most recent estimates from the Gaia mission suggest that there are approximately 100 to 400 billion stars in the Milky Way, with significant uncertainties remaining due to the difficulty in detecting faint red dwarfs and brown dwarfs."\
    },\
    "delta": { "role": "assistant", "content": "" }\
  }\
]

```

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-usage)

usage

object

Usage statistics for the completion request.

Showchild attributes

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-usage-prompt-tokens)

usage.prompt\_tokens

integer

The number of tokens provided in the request prompt.

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-usage-completion-tokens)

usage.completion\_tokens

integer

The number of tokens generated in the response output.

[​](https://docs.perplexity.ai/api-reference/chat-completions#response-usage-total-tokens)

usage.total\_tokens

integer

The total number of tokens used in the chat completion (prompt + completion).

cURL

Python

JavaScript

PHP

Go

Java

Copy

```
curl --request POST \
  --url https://api.perplexity.ai/chat/completions \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: application/json' \
  --data '{
  "model": "sonar",
  "messages": [\
    {\
      "role": "system",\
      "content": "Be precise and concise."\
    },\
    {\
      "role": "user",\
      "content": "How many stars are there in our galaxy?"\
    }\
  ],
  "max_tokens": 123,
  "temperature": 0.2,
  "top_p": 0.9,
  "search_domain_filter": null,
  "return_images": false,
  "return_related_questions": false,
  "search_recency_filter": "<string>",
  "top_k": 0,
  "stream": false,
  "presence_penalty": 0,
  "frequency_penalty": 1,
  "response_format": null
}'
```

200

400

401

429

500

504

524

Copy

```
{
  "id": "3c90c3cc-0d44-4b50-8888-8dd25736052a",
  "model": "sonar",
  "object": "chat.completion",
  "created": 1724369245,
  "citations": [\
    "https://www.astronomy.com/science/astro-for-kids-how-many-stars-are-there-in-space/",\
    "https://www.esa.int/Science_Exploration/Space_Science/Herschel/How_many_stars_are_there_in_the_Universe",\
    "https://www.space.com/25959-how-many-stars-are-in-the-milky-way.html",\
    "https://www.space.com/26078-how-many-stars-are-there.html",\
    "https://en.wikipedia.org/wiki/Milky_Way"\
  ],
  "choices": [\
    {\
      "index": 0,\
      "finish_reason": "stop",\
      "message": {\
        "role": "assistant",\
        "content": "The number of stars in the Milky Way galaxy is estimated to be between 100 billion and 400 billion stars. The most recent estimates from the Gaia mission suggest that there are approximately 100 to 400 billion stars in the Milky Way, with significant uncertainties remaining due to the difficulty in detecting faint red dwarfs and brown dwarfs."\
      },\
      "delta": {\
        "role": "assistant",\
        "content": ""\
      }\
    }\
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 70,
    "total_tokens": 84
  }
}
```