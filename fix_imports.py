import os

def rep(p, d):
    if not os.path.exists(p):
        print(f"Skipping {p}, does not exist")
        return
    with open(p, 'r') as f:
        c = f.read()
    for k, v in d.items():
        c = c.replace(k, v)
    with open(p, 'w') as f:
        f.write(c)

rep('server/environment.py', {
    'from env.models import': 'from models import',
    'from env.demand_generator import': 'from server.demand_generator import',
    'from env.disruption import': 'from server.disruption import',
    'env/tasks/': 'server/tasks/'
})

rep('server/graders/composite_grader.py', {
    'from env.models import EpisodeResult': 'from models import EpisodeResult',
    'from graders.cost_grader import': 'from server.graders.cost_grader import',
    'from graders.service_grader import': 'from server.graders.service_grader import'
})

rep('server/app.py', {
    'from env.environment import': 'from server.environment import',
    'from env.models import': 'from models import',
    'from graders.composite_grader import': 'from server.graders.composite_grader import'
})

rep('tests/test_environment.py', {
    'from env.environment import': 'from server.environment import',
    'from env.models import': 'from models import',
    'from graders.composite_grader import': 'from server.graders.composite_grader import'
})

rep('tests/test_graders.py', {
    'from env.models import': 'from models import',
    'from graders.': 'from server.graders.'
})

rep('inference.py', {
    'from env.environment import': 'from server.environment import',
    'from env.models import': 'from models import',
    'from graders.composite_grader import': 'from server.graders.composite_grader import'
})
