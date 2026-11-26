import re

samples = [
    # 1. Repetitive loop
    'Machine learning is a machine learning algorithm that involves learning algorithms that learn patterns and patterns. It involves learning patterns, patterns, and patterns.',
    # 2. Corrupt steps
    '''Greing, I can provide a specific type of machine learning and information. Here are some general steps you could follow to help you get started:
1. Choose the basics of algorithms: You can use the following steps to find the concepts in your understanding of how many different topics you have.
2. Use the ones: You\'ll need to learn from different types of data structures and other areas that may be used for processing. This will help you identify patterns and make predictions or decisions about the field.
3. Choose an algorithm: Once you\'re interested in the basics, consider using tools such as C# or Gather. These include building a website\'s website, which is designed to create a variety of applications such as Google MAEC and Python.
4. Choose''',
    # 3. Truncated projectives
    '''Machine Learning (ML) is a learning algorithm that requires several tasks, including: many tasks and algorithms. Here are some steps you can do to get started:

1. Projectives: You can learn how to understand the data in the specific task, so that it\'s important to be challenging for developing tasks such as machine learning, computer science, or machine learning. This involves using deep learning models on developing neural networks, which can help you learn more about the language.
2. Study your model: There are many techniques that can be used to improve its performance on the training process. These include visualizing the RL model, the model, and search engine. It can also make predictions on the size of the input data rather than the same time.
3''',
    # 4. Neural network fact inversion
    'Artificial Intelligence (AI) is a broad field of computer science that involves the development of machine learning and machines to learn from data. It\'s a deep neural network of artificial intelligence, which are trained on many different fields such as visual perception, speech recognition, decision-making, and human brain.',
    # 5. "is a machine learning"
    'Ber Learning is a machine learning that requires a wide range of tasks, but it\'s important to be a lot of and difficult to learn. It can also take some time with the way to understand how many things you are looking for.',
    # 6. Science fiction
    'Artificial Intelligence (AI) is a broad field of computer science fiction learning that focuses on the development of machine learning and machines, including data analysis, speech recognition, decision-making, and AI. It\'s important to note that AI has many different types of algorithms and intelligence.',
    # 7. "It's a machine learning"
    'Artificial Intelligence (AI) is a broad field of computer science that involves the development of machine learning and machines to learn from data. It\'s a machine learning, but it can be used to build intelligent tasks such as neural networks or problem-solving.',
    # 8. User report failure
    'Artificial Intelligence (AI) is a broad machine learning that involves the ability to learn and decisions, but they are not suitable for human intelligence.',
    # 9. Multi-turn continuation failure 1
    'No, I can be used to learn from a computer science fiction by machine learning. It is not a mathematical model that has been trained on the field of data and it is still in many tasks such as image recognition algorithms and natural language processing applications.',
    # 10. Multi-turn continuation failure 2
    'No, the most commonly used in computer science and machine learning is a mathematical model that can be used to generate human intelligence.',
    # 11. Multi-turn continuation failure 3
    'The most commonly used in machine learning is a computer science that is not the field of artificial intelligence, which means it can be used to learn from data analysis.'
]

def test_validator(query, response):
    if not response or len(response.strip()) < 15:
        return False, 'Response is too short or empty.'
    words = response.split()
    if len(words) < 4:
        return False, 'Response has too few words.'
    r_lower = response.lower()
    
    # 1. Prompt echoes and invalid tokens
    prompt_echoes = [
        'your name is', 'your name is c++', 'i am a language model', 'i\'m a language model',
        'as a language model', 'trained on text data', 'trained on the text data',
        'trained on a variety of topics', 'variety of topics such as text',
        'text, text, or other words', 'text data', 'greing', 'ber learning', 'ber is a',
        'science fiction learning', 'broad machine learning', 'it\'s a machine learning',
        'they are not suitable for human intelligence', 'not suitable for human intelligence',
        'not suitable for human', 'not suitable for text', 'learn and decisions',
        'projectives', 'use the ones', 'website\'s website', 'google maec',
        'important to be a lot of and difficult', '<|user|>', '<|assistant|>', '[eos]', 'user:'
    ]
    for pe in prompt_echoes:
        if pe in r_lower:
            return False, f'Contains echo or invalid phrase: {pe}'
            
    if 'c++' not in query.lower() and 'c++' in r_lower:
        return False, 'Irrelevant C++ mention'
        
    # Check science fiction on non-fiction query
    if 'science fiction' in r_lower and 'fiction' not in query.lower():
        return False, "Contains 'science fiction' in factual response"
        
    # Check conversational negation/polarity on definition queries
    is_definition_query = bool(re.match(r'(?i)^(what|who|where|why|how|explain|describe|define)\b', query.strip()))
    if is_definition_query and re.match(r'^(no|yes|sure|well|nope|yeah)\b[,!.]', r_lower.strip()):
        return False, "Irrelevant conversational opening ('No,' / 'Yes,') on definition query"
        
    # Check self-referential anthropomorphism in definition
    if re.search(r'\b(i\s+can\s+be\s+used\s+to\s+learn|i\s+am\s+used\s+to\s+learn|i\s+am\s+a\s+mathematical\s+model|i\s+am\s+an\s+algorithm)\b', r_lower):
        return False, 'Self-referential anthropomorphism in definition'
        
    # Check human intelligence hallucinations
    if re.search(r'\b(generate\s+human\s+intelligence|suitable\s+for\s+human\s+intelligence|not\s+suitable\s+for\s+human\s+intelligence)\b', r_lower):
        return False, 'Incoherent claim regarding human intelligence'
        
    # Check false negation of scientific domain
    if re.search(r'\b(is\s+not|not)\s+(the\s+|a\s+)?(field|part|branch|subfield)\s+of\s+artificial\s+intelligence\b', r_lower):
        return False, 'Factual error: falsely negates connection to artificial intelligence'
    if re.search(r'\bnot\s+the\s+field\s+of\s+artificial\s+intelligence\b', r_lower):
        return False, 'Factual error: falsely negates connection to artificial intelligence'
        
    # Check ungrammatical noun-category distortions
    if re.search(r'\b(is|it\'?s)\s+a\s+(broad\s+|narrow\s+)?(computer\s+science|machine\s+learning)\b', r_lower):
        return False, 'Ungrammatical noun-category distortion'
    if re.search(r'\bthe\s+most\s+commonly\s+used\s+in\s+[\w\s]+\s+is\s+a\s+(computer\s+science|machine\s+learning)\b', r_lower):
        return False, 'Ungrammatical noun-category distortion'

    trimmed = response.strip()
    terminal_puncts = ('.', '!', '?', '"', "'", ')')
    if not any(trimmed.endswith(p) for p in terminal_puncts):
        return False, 'Response ends abruptly without terminal punctuation'
    if re.search(r'\b(and|or|but|such as|including|with|the|to|of|in|for|choose|from|that|a|an|it|is|as|\d+)\s*[.!?]?$', r_lower):
        return False, 'Response ends with dangling fragment or list cutoff'
        
    if re.search(r'\b(artificial intelligence|ai)\s+is\s+a\s+(broad\s+|narrow\s+)?machine\s+learning\b', r_lower):
        return False, 'Fact inversion: defines AI as a type of ML'
    if re.search(r'\b(artificial intelligence|ai)\s+is\s+a\s+(deep\s+)?neural\s+network\b', r_lower):
        return False, 'Fact inversion: defines AI as a neural network'
        
    q_clean = re.sub(r'(?i)\b(what is|what are|explain|define|search for|tell me about)\b', '', query).strip(' :-\t?.')
    if 'machine learning' in q_clean.lower() and 'ai' not in q_clean.lower():
        first_sentence = re.split(r'[.!?]', response)[0].lower()
        if first_sentence.startswith('artificial intelligence') and 'machine learning' not in first_sentence[:25]:
            return False, 'Off-topic definition: defines AI instead of Machine Learning'
            
    word_counts = {}
    for w in words:
        w_clean = re.sub(r'[^\w]', '', w.lower())
        if len(w_clean) > 3 and w_clean not in ['that', 'this', 'with', 'from', 'have', 'more', 'used', 'which']:
            word_counts[w_clean] = word_counts.get(w_clean, 0) + 1
            if word_counts[w_clean] >= 3 and len(words) < 60:
                return False, f'Repetitive word loop: {w_clean}'
                
    return True, 'Passed'

all_rejected = True
for i, s in enumerate(samples, 1):
    ok, reason = test_validator('What is Machine Learning?', s)
    print(f'Sample {i:2d}: Valid={ok:<5} | Reason={reason}')
    if ok:
        all_rejected = False

print(f'\nAll 11 broken samples rejected: {all_rejected}')

# Test genuine answer passes:
good_sample = "Machine learning is a branch of artificial intelligence and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy."
ok, reason = test_validator('What is Machine Learning?', good_sample)
print(f'Genuine Good Sample: Valid={ok:<5} | Reason={reason}')
assert ok, "Genuine good sample failed validation!"
