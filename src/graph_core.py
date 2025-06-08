#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Core pour l'Analyse de Graphes
Université de Yaoundé I - Département d'Informatique

Auteurs: DIZE TCHEMOU MIGUEL CAREY, SAGUEN KAMDEM CHERYL RONALD, 
         SIGNE FONGANG WILFRIED BRANDON
Supervisé par: Dr Manga MAXWELL
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque, defaultdict
import math

class GraphAnalyzer:
    """Classe principale pour l'analyse des graphes"""
    
    def __init__(self):
        """Initialise l'analyseur de graphes"""
        self.graph_data = None
        self.distance_matrix = None
    
    def parse_adjacency_matrix(self, matrix_text):
        """
        Parse une matrice d'adjacence depuis un texte
        
        Args:
            matrix_text (str): Texte contenant la matrice
            
        Returns:
            numpy.ndarray: Matrice d'adjacence
        """
        try:
            lines = matrix_text.strip().split('\n')
            matrix = []
            
            for line in lines:
                row = [int(x) for x in line.strip().split()]
                matrix.append(row)
            
            matrix = np.array(matrix)
            
            # Vérification de la validité
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("La matrice doit être carrée")
            
            if not np.all((matrix == 0) | (matrix == 1)):
                raise ValueError("La matrice ne doit contenir que des 0 et des 1")
            
            # Éliminer les boucles pour le calcul du diamètre
            np.fill_diagonal(matrix, 0)
            
            return matrix
            
        except Exception as e:
            raise ValueError(f"Erreur lors du parsing de la matrice: {str(e)}")
    
    def parse_adjacency_list(self, list_text):
        """
        Parse une liste d'adjacence depuis un texte
        
        Args:
            list_text (str): Texte contenant la liste d'adjacence
            
        Returns:
            dict: Dictionnaire représentant la liste d'adjacence
        """
        try:
            lines = list_text.strip().split('\n')
            adj_dict = {}
            
            for line in lines:
                if ':' not in line:
                    continue
                    
                parts = line.split(':')
                vertex = int(parts[0].strip())
                
                # Gestion des listes vides
                neighbors_str = parts[1].strip()
                if not neighbors_str:
                    neighbors = []
                else:
                    neighbors = [int(x.strip()) for x in neighbors_str.split(',') if x.strip()]
                
                adj_dict[vertex] = neighbors
            
            return adj_dict
            
        except Exception as e:
            raise ValueError(f"Erreur lors du parsing de la liste: {str(e)}")
    
    def adjacency_list_to_matrix(self, adj_dict):
        """
        Convertit une liste d'adjacence en matrice d'adjacence
        
        Args:
            adj_dict (dict): Liste d'adjacence
            
        Returns:
            numpy.ndarray: Matrice d'adjacence
        """
        if not adj_dict:
            raise ValueError("Liste d'adjacence vide")
        
        # Trouver le nombre maximum de sommets
        max_vertex = max(max(adj_dict.keys()), 
                        max([max(neighbors) for neighbors in adj_dict.values() if neighbors], 
                            default=0))
        
        size = max_vertex + 1
        matrix = np.zeros((size, size), dtype=int)
        
        # Remplir la matrice
        for vertex, neighbors in adj_dict.items():
            for neighbor in neighbors:
                matrix[vertex][neighbor] = 1
                # Pour un graphe non orienté, ajouter l'arête symétrique
                matrix[neighbor][vertex] = 1
        
        return matrix
    
    def is_connected(self, matrix):
        """
        Vérifie si le graphe est connexe en utilisant DFS
        
        Args:
            matrix (numpy.ndarray): Matrice d'adjacence
            
        Returns:
            bool: True si le graphe est connexe, False sinon
        """
        n = len(matrix)
        if n == 0:
            return True
        
        visited = [False] * n
        
        # Commencer DFS depuis le premier sommet
        self._dfs(matrix, 0, visited)
        
        # Vérifier si tous les sommets ont été visités
        return all(visited)
    
    def _dfs(self, matrix, vertex, visited):
        """
        Parcours en profondeur récursif
        
        Args:
            matrix (numpy.ndarray): Matrice d'adjacence
            vertex (int): Sommet courant
            visited (list): Liste des sommets visités
        """
        visited[vertex] = True
        
        # Visiter tous les voisins non visités
        for neighbor in range(len(matrix)):
            if matrix[vertex][neighbor] == 1 and not visited[neighbor]:
                self._dfs(matrix, neighbor, visited)
    
    def floyd_warshall(self, matrix):
        """
        Implémente l'algorithme Floyd-Warshall pour trouver toutes les distances
        
        Args:
            matrix (numpy.ndarray): Matrice d'adjacence
            
        Returns:
            numpy.ndarray: Matrice des distances
        """
        n = len(matrix)
        dist = np.full((n, n), float('inf'))
        
        # Initialisation: distance = 0 pour la diagonale, 1 pour les arêtes directes
        for i in range(n):
            dist[i][i] = 0
            for j in range(n):
                if matrix[i][j] == 1:
                    dist[i][j] = 1
        
        # Algorithme Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        self.distance_matrix = dist
        return dist
    
    def calculate_diameter(self, matrix):
        """
        Calcule le diamètre du graphe
        
        Args:
            matrix (numpy.ndarray): Matrice d'adjacence
            
        Returns:
            float: Diamètre du graphe (inf si non connexe)
        """
        if not self.is_connected(matrix):
            return float('inf')
        
        dist_matrix = self.floyd_warshall(matrix)
        
        # Le diamètre est la plus grande distance finie
        finite_distances = dist_matrix[dist_matrix != float('inf')]
        
        if len(finite_distances) == 0:
            return float('inf')
        
        # Exclure les distances nulles (diagonale)
        non_zero_distances = finite_distances[finite_distances > 0]
        
        if len(non_zero_distances) == 0:
            return 0
        
        return int(np.max(non_zero_distances))
    
    def get_connected_components(self, matrix):
        """
        Trouve toutes les composantes connexes du graphe
        
        Args:
            matrix (numpy.ndarray): Matrice d'adjacence
            
        Returns:
            list: Liste des composantes connexes
        """
        n = len(matrix)
        visited = [False] * n
        components = []
        
        for i in range(n):
            if not visited[i]:
                component = []
                self._dfs_component(matrix, i, visited, component)
                components.append(component)
        
        return components
    
    def _dfs_component(self, matrix, vertex, visited, component):
        """
        DFS pour trouver une composante connexe
        
        Args:
            matrix (numpy.ndarray): Matrice d'adjacence
            vertex (int): Sommet courant
            visited (list): Liste des sommets visités
            component (list): Composante courante
        """
        visited[vertex] = True
        component.append(vertex)
        
        for neighbor in range(len(matrix)):
            if matrix[vertex][neighbor] == 1 and not visited[neighbor]:
                self._dfs_component(matrix, neighbor, visited, component)
    
    def is_symmetric(self, matrix):
        """
        Vérifie si la matrice est symétrique (graphe non orienté)
        
        Args:
            matrix (numpy.ndarray): Matrice d'adjacence
            
        Returns:
            bool: True si symétrique, False sinon
        """
        return np.array_equal(matrix, matrix.T)
    
    def get_graph_properties(self, matrix):
        """
        Calcule diverses propriétés du graphe
        
        Args:
            matrix (numpy.ndarray): Matrice d'adjacence
            
        Returns:
            dict: Dictionnaire des propriétés
        """
        n = len(matrix)
        edges = np.sum(matrix) // 2 if self.is_symmetric(matrix) else np.sum(matrix)
        
        # Degrés des sommets
        degrees = np.sum(matrix, axis=1)
        
        # Composantes connexes
        components = self.get_connected_components(matrix)
        
        properties = {
            'vertices': n,
            'edges': int(edges),
            'is_directed': not self.is_symmetric(matrix),
            'is_connected': self.is_connected(matrix),
            'diameter': self.calculate_diameter(matrix),
            'components': len(components),
            'component_sizes': [len(comp) for comp in components],
            'degrees': degrees.tolist(),
            'min_degree': int(np.min(degrees)),
            'max_degree': int(np.max(degrees)),
            'avg_degree': float(np.mean(degrees))
        }
        
        return properties
    
    def visualize_graph(self, matrix, ax=None):
        """
        Visualise le graphe avec NetworkX et Matplotlib
        
        Args:
            matrix (numpy.ndarray): Matrice d'adjacence
            ax (matplotlib.axes): Axes pour le plot (optionnel)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            ax.clear()
        
        # Créer le graphe NetworkX
        G = nx.from_numpy_array(matrix)
        
        # Définir la disposition des nœuds
        if len(matrix) <= 10:
            pos = nx.spring_layout(G, k=2, iterations=50)
        else:
            pos = nx.spring_layout(G, k=1, iterations=30)
        
        # Couleurs selon la connexité
        if self.is_connected(matrix):
            node_colors = ['#2ecc71'] * len(matrix)  # Vert pour connexe
            title_color = '#27ae60'
        else:
            # Différentes couleurs pour chaque composante
            components = self.get_connected_components(matrix)
            colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c', 
                     '#e67e22', '#95a5a6', '#34495e']
            node_colors = ['#bdc3c7'] * len(matrix)  # Gris par défaut
            
            for i, component in enumerate(components):
                color = colors[i % len(colors)]
                for node in component:
                    node_colors[node] = color
            title_color = '#c0392b'
        
        # Dessiner les nœuds
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=800, alpha=0.9)
        
        # Dessiner les arêtes
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#7f8c8d', 
                              width=2, alpha=0.6)
        
        # Dessiner les labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, 
                               font_weight='bold', font_color='white')
        
        # Titre et informations
        diameter = self.calculate_diameter(matrix)
        diameter_str = str(diameter) if diameter != float('inf') else "∞"
        
        title = f"Graphe - {len(matrix)} sommets, {np.sum(matrix)//2} arêtes\n"
        title += f"{'Connexe' if self.is_connected(matrix) else 'Non Connexe'} | "
        title += f"Diamètre: {diameter_str}"
        
        ax.set_title(title, fontsize=14, fontweight='bold', color=title_color, pad=20)
        ax.axis('off')
        
        # Légende pour les composantes si non connexe
        if not self.is_connected(matrix):
            components = self.get_connected_components(matrix)
            legend_elements = []
            colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']
            
            for i, component in enumerate(components):
                color = colors[i % len(colors)]
                legend_elements.append(
                    patches.Patch(color=color, 
                                label=f'Composante {i+1}: {len(component)} sommets')
                )
            
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
    
    def get_shortest_paths(self, matrix):
        """
        Trouve tous les plus courts chemins entre les paires de sommets
        
        Args:
            matrix (numpy.ndarray): Matrice d'adjacence
            
        Returns:
            dict: Dictionnaire des plus courts chemins
        """
        n = len(matrix)
        dist_matrix = self.floyd_warshall(matrix)
        paths = {}
        
        # Reconstruction des chemins avec Floyd-Warshall modifié
        next_vertex = np.full((n, n), -1, dtype=int)
        
        # Initialisation
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i][j] == 1:
                    next_vertex[i][j] = j
        
        # Floyd-Warshall pour la reconstruction des chemins
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if (dist_matrix[i][k] + dist_matrix[k][j] < dist_matrix[i][j]):
                        next_vertex[i][j] = next_vertex[i][k]
        
        # Construire les chemins
        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i][j] != float('inf'):
                    path = self._reconstruct_path(i, j, next_vertex)
                    paths[(i, j)] = path
        
        return paths
    
    def _reconstruct_path(self, start, end, next_vertex):
        """
        Reconstruit un chemin à partir de la matrice next_vertex
        
        Args:
            start (int): Sommet de départ
            end (int): Sommet d'arrivée
            next_vertex (numpy.ndarray): Matrice des prochains sommets
            
        Returns:
            list: Chemin du sommet start au sommet end
        """
        if next_vertex[start][end] == -1:
            return []
        
        path = [start]
        current = start
        
        while current != end:
            current = next_vertex[current][end]
            path.append(current)
        
        return path
    
    def analyze_centrality(self, matrix):
        """
        Calcule différentes mesures de centralité
        
        Args:
            matrix (numpy.ndarray): Matrice d'adjacence
            
        Returns:
            dict: Mesures de centralité
        """
        G = nx.from_numpy_array(matrix)
        
        centrality = {
            'degree': dict(nx.degree_centrality(G)),
            'closeness': dict(nx.closeness_centrality(G)) if self.is_connected(matrix) else {},
            'betweenness': dict(nx.betweenness_centrality(G)),
            'eigenvector': dict(nx.eigenvector_centrality(G, max_iter=1000)) if self.is_connected(matrix) else {}
        }
        
        return centrality
    
    def detect_cycles(self, matrix):
        """
        Détecte les cycles dans le graphe
        
        Args:
            matrix (numpy.ndarray): Matrice d'adjacence
            
        Returns:
            list: Liste des cycles trouvés
        """
        G = nx.from_numpy_array(matrix)
        
        try:
            cycles = list(nx.simple_cycles(G))
            return cycles
        except:
            # Pour les graphes non orientés, utiliser une approche différente
            cycles = []
            try:
                cycle_basis = nx.cycle_basis(G)
                cycles = cycle_basis
            except:
                pass
            
            return cycles
    
    def export_to_formats(self, matrix, filename_base):
        """
        Exporte le graphe dans différents formats
        
        Args:
            matrix (numpy.ndarray): Matrice d'adjacence
            filename_base (str): Nom de base pour les fichiers
        """
        G = nx.from_numpy_array(matrix)
        
        # Export en GEXF (Gephi)
        try:
            nx.write_gexf(G, f"{filename_base}.gexf")
        except:
            pass
        
        # Export en GraphML
        try:
            nx.write_graphml(G, f"{filename_base}.graphml")
        except:
            pass
        
        # Export en format DOT (Graphviz)
        try:
            nx.nx_agraph.write_dot(G, f"{filename_base}.dot")
        except:
            pass
