import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import platform

if platform.node() == 'Nick_Laptop':
    drive = 'C'
elif platform.node() == 'MSI':
    drive = 'D'
else:
    drive = 'uhhhhhh'
    print('Uhhhhhhhhhhhhh')
os.chdir(f'{drive}:/PhD/DissolutionProgramming/REB---Rebellion-Paper/')

# Create a directed acyclic graph
dag = nx.DiGraph()

# Add edges
dag.add_edges_from([('Population', 'Wealth'),
                    ('Wealth', 'Monastic\nLand'),
                    ('Wealth', 'Tax\nChanges'),
                    ('Monastic\nLand', 'Rebel\nMuster'),
                    ('Tax\nChanges', 'Rebel\nMuster'),
                    ('Population', 'Rebel\nMuster'),
                    ('Population', 'Monastic\nLand'),
                    ('Wealth', 'Rebel\nMuster'),
                    ('Parish\nReligion', 'Rebel\nMuster'),
                    ])

# Draw the graph
plt.figure(figsize=(10,10))
posdict = {'Rebel\nMuster': (1, -.5),
           'Population': (0, .9),
           'Wealth': (-1, -.5),
           'Monastic\nLand': (0, -1),
           'Tax\nChanges': (-1, .5),
           'Parish\nReligion': (1, .5)}
nx.draw(dag,
        pos=posdict,
        node_shape='s',
        node_size=8000,
        node_color= ['skyblue',
                     'skyblue',
                     'palegreen',
                     'skyblue',
                     'orangered',
                     'skyblue',
                     ],
        with_labels=True)
plt.savefig(r'Output\Images\Graphs\LittleRebellionDAG.png')
plt.show()

#%% Bigger Rebellion DAG

bigDag = nx.DiGraph()
bigDag.add_edges_from([('Population', 'Wealth'),
                    ('Wealth', 'Monastic\nLand'),
                    ('Wealth', 'Tax\nChanges'),
                    ('Monastic\nLand', 'Rebel\nMuster'),
                    ('Tax\nChanges', 'Rebel\nMuster'),
                    ('Population', 'Rebel\nMuster'),
                    ('Population', 'Monastic\nLand'),
                    ('Parish Religion Changes', 'Rebel\nMuster'),
                    ('Population', 'Tithe'),
                    ('Wealth', 'Tithe'),
                    ('Population', 'Alms'),
                    ('Wealth', 'Alms'),
                    ('Monastic Net Income', 'Alms'),
                    ('Tithe', 'Rebel\nMuster'),
                    ('Alms', 'Rebel\nMuster'),
                    ('Monastic Net Income', 'Rebel\nMuster')]
                      )
nx.draw_networkx(bigDag,
        node_shape='s',
        node_color='skyblue',
        with_labels=True)

plt.savefig('Output/Images/Graphs/BigRebellionDAG.png')
plt.show()

#%% Bigger Dag

biggerDag = nx.DiGraph()
biggerDag.add_edges_from([('Population', 'Wealth'),
                          ('Wealth', 'Monastic\nLand'),
                          ('Wealth', 'Tax\nChanges'),
                          ('Monastic\nLand', 'Rebel\nMuster'),
                          ('Tax\nChanges', 'Rebel\nMuster'),
                          ('Population', 'Rebel\nMuster'),
                          ('Population', 'Monastic\nLand'),
                          ('Parish Religion Changes', 'Rebel\nMuster'),
                          ('Population', 'Tithe'),
                          ('Wealth', 'Tithe'),
                          ('Population', 'Alms'),
                          ('Wealth', 'Alms'),
                          ('Monastic Net Income', 'Alms'),
                          ('Tithe', 'Rebel\nMuster'),
                          ('Alms', 'Rebel\nMuster'),
                          ('Monastic Net Income', 'Rebel\nMuster'),
                          ('Area', 'Population'),
                          ('Area', 'Wealth'),
                          ('Terrain', 'Wealth'),
                          ('Terrain', 'Area'),

                          ]
                      )
nx.draw_networkx(biggerDag,
                 node_shape='s',
                 node_color='skyblue',
                 with_labels=True
                 )
plt.show()