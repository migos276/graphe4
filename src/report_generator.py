#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Générateur de Rapports HTML pour l'Analyse de Graphes
Université de Yaoundé I - Département d'Informatique

Auteurs: DIZE TCHEMOU MIGUEL CAREY, SAGUEN KAMDEM CHERYL RONALD, 
         SIGNE FONGANG WILFRIED BRANDON
Supervisé par: Dr Manga MAXWELL
"""

import os
import datetime
import numpy as np
from graph_core import GraphAnalyzer
import base64
import io
import matplotlib.pyplot as plt

class ReportGenerator:
    """Générateur de rapports HTML académiques"""
    
    def __init__(self):
        """Initialise le générateur de rapports"""
        self.analyzer = GraphAnalyzer()
    
    def generate_report(self, graph_data):
        """
        Génère un rapport HTML complet
        
        Args:
            graph_data (dict): Données du graphe analysé
            
        Returns:
            str: Chemin du fichier généré
        """
        matrix = graph_data['data']
        properties = self.analyzer.get_graph_properties(matrix)
        
        # Génération du graphique
        graph_image = self._generate_graph_image(matrix)
        
        # Construction du HTML
        html_content = self._build_html_report(graph_data, properties, graph_image)
        
        # Sauvegarde
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rapport_analyse_graphe_{timestamp}.html"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename
    
    def _generate_graph_image(self, matrix):
        """
        Génère une image du graphe en base64
        
        Args:
            matrix (numpy.ndarray): Matrice d'adjacence
            
        Returns:
            str: Image encodée en base64
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        self.analyzer.visualize_graph(matrix, ax)
        
        # Conversion en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return image_base64
    
    def _build_html_report(self, graph_data, properties, graph_image):
        """
        Construit le contenu HTML du rapport
        
        Args:
            graph_data (dict): Données du graphe
            properties (dict): Propriétés calculées
            graph_image (str): Image du graphe en base64
            
        Returns:
            str: Contenu HTML complet
        """
        matrix = graph_data['data']
        
        # Calculs supplémentaires
        distance_matrix = self.analyzer.floyd_warshall(matrix)
        components = self.analyzer.get_connected_components(matrix)
        
        html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport d'Analyse de Graphe - Connexité et Diamètre</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        {self._generate_header()}
        {self._generate_summary(graph_data, properties)}
        {self._generate_methodology()}
        {self._generate_results(matrix, properties, distance_matrix, components)}
        {self._generate_visualization(graph_image)}
        {self._generate_technical_details(matrix, properties)}
        {self._generate_conclusion(properties)}
        {self._generate_appendix(matrix, distance_matrix)}
        {self._generate_footer()}
    </div>
</body>
</html>
        """
        
        return html
    
    def _get_css_styles(self):
        """Retourne les styles CSS pour le rapport"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            color: #333;
            background-color: #fff;
        }
        
        .container {
            max-width: 210mm;
            margin: 0 auto;
            padding: 20mm;
            background: white;
            min-height: 297mm;
        }
        
        .header {
            text-align: center;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .university-name {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        
        .department {
            font-size: 18px;
            color: #34495e;
            margin-bottom: 5px;
        }
        
        .level {
            font-size: 16px;
            color: #7f8c8d;
            margin-bottom: 15px;
        }
        
        .title {
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
            margin: 20px 0;
            text-transform: uppercase;
        }
        
        .supervisor {
            font-size: 14px;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .authors {
            font-size: 14px;
            color: #34495e;
        }
        
        .section {
            margin: 30px 0;
            page-break-inside: avoid;
        }
        
        .section h2 {
            color: #2c3e50;
            font-size: 18px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
            margin-bottom: 15px;
        }
        
        .section h3 {
            color: #34495e;
            font-size: 16px;
            margin: 20px 0 10px 0;
        }
        
        .matrix-container {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            overflow-x: auto;
        }
        
        .matrix {
            font-family: 'Courier New', monospace;
            font-size: 14px;
            white-space: pre;
            text-align: center;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        
        .result-box {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        
        .result-label {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        
        .result-value {
            font-size: 18px;
            color: #27ae60;
            font-weight: bold;
        }
        
        .result-value.negative {
            color: #e74c3c;
        }
        
        .graph-image {
            text-align: center;
            margin: 20px 0;
        }
        
        .graph-image img {
            max-width: 100%;
            height: auto;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
        }
        
        .algorithm-box {
            background: #f4f6f7;
            border: 1px solid #d5dbdb;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .complexity {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 10px;
            border-radius: 3px;
            margin: 10px 0;
        }
        
        .footer {
            border-top: 2px solid #2c3e50;
            padding-top: 15px;
            margin-top: 40px;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        
        th, td {
            border: 1px solid #bdc3c7;
            padding: 8px;
            text-align: center;
        }
        
        th {
            background-color: #34495e;
            color: white;
            font-weight: bold;
        }
        
        .highlight {
            background-color: #f1c40f;
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        .page-break {
            page-break-before: always;
        }
        
        @media print {
            .container {
                max-width: none;
                margin: 0;
                padding: 15mm;
            }
            
            .section {
                page-break-inside: avoid;
            }
        }
        """
    
    def _generate_header(self):
        """Génère l'en-tête du rapport"""
        current_date = datetime.datetime.now().strftime("%d/%m/%Y")
        
        return f"""
        <div class="header">
            <div class="university-name">UNIVERSITÉ DE YAOUNDÉ I</div>
            <div class="department">DÉPARTEMENT D'INFORMATIQUE</div>
            <div class="level">Département d'Informatique - Licence 2</div>
            
            <div class="title">
                Rapport d'Analyse de Graphe:<br>
                Vérification de Connexité et Calcul du Diamètre
            </div>
            
            <div class="supervisor">
                <strong>Supervisé par:</strong> Dr Manga MAXWELL
            </div>
            
            <div class="authors">
                <strong>Membres du groupe:</strong><br>
                • DIZE TCHEMOU MIGUEL CAREY<br>
                • SAGUEN KAMDEM CHERYL RONALD<br>
                • SIGNE FONGANG WILFRIED BRANDON
            </div>
            
            <div style="margin-top: 15px; font-size: 14px;">
                <strong>Date:</strong> {current_date}
            </div>
        </div>
        """
    
    def _generate_summary(self, graph_data, properties):
        """Génère le résumé exécutif"""
        diameter_str = str(properties['diameter']) if properties['diameter'] != float('inf') else "∞"
        
        return f"""
        <div class="section">
            <h2>1. RÉSUMÉ EXÉCUTIF</h2>
            <p>
                Ce rapport présente l'analyse complète d'un graphe comportant <strong>{properties['vertices']}</strong> 
                sommets et <strong>{properties['edges']}</strong> arêtes. L'analyse se concentre sur deux aspects 
                fondamentaux de la théorie des graphes : la <em>connexité</em> et le <em>diamètre</em>.
            </p>
            
            <div class="results-grid">
                <div class="result-box">
                    <div class="result-label">État de Connexité</div>
                    <div class="result-value {'positive' if properties['is_connected'] else 'negative'}">
                        {'CONNEXE' if properties['is_connected'] else 'NON CONNEXE'}
                    </div>
                </div>
                
                <div class="result-box">
                    <div class="result-label">Diamètre du Graphe</div>
                    <div class="result-value">
                        {diameter_str}
                    </div>
                </div>
                
                <div class="result-box">
                    <div class="result-label">Nombre de Composantes</div>
                    <div class="result-value">
                        {properties['components']}
                    </div>
                </div>
                
                <div class="result-box">
                    <div class="result-label">Type de Graphe</div>
                    <div class="result-value">
                        {'ORIENTÉ' if properties['is_directed'] else 'NON ORIENTÉ'}
                    </div>
                </div>
            </div>
            
            <p>
                {'Le graphe analysé est connexe, ce qui signifie qu\'il existe un chemin entre toute paire de sommets. Le diamètre calculé représente la plus grande distance géodésique entre deux sommets quelconques du graphe.' if properties['is_connected'] else f'Le graphe analysé n\'est pas connexe et se compose de {properties["components"]} composantes connexes distinctes. En raison de cette non-connexité, le diamètre est considéré comme infini.'}
            </p>
        </div>
        """
    
    def _generate_methodology(self):
        """Génère la section méthodologie"""
        return """
        <div class="section">
            <h2>2. MÉTHODOLOGIE ET ALGORITHMES</h2>
            
            <h3>2.1 Vérification de la Connexité</h3>
            <div class="algorithm-box">
                <p><strong>Algorithme utilisé:</strong> Parcours en Profondeur (DFS - Depth-First Search)</p>
                <p><strong>Principe:</strong> L'algorithme effectue un parcours en profondeur à partir d'un sommet arbitraire. 
                Si tous les sommets du graphe sont visités, alors le graphe est connexe.</p>
                
                <div class="complexity">
                    <strong>Complexité temporelle:</strong> O(V + E) où V est le nombre de sommets et E le nombre d'arêtes<br>
                    <strong>Complexité spatiale:</strong> O(V) pour la pile de récursion et le tableau des visités
                </div>
            </div>
            
            <h3>2.2 Calcul du Diamètre</h3>
            <div class="algorithm-box">
                <p><strong>Algorithme utilisé:</strong> Floyd-Warshall</p>
                <p><strong>Principe:</strong> Cet algorithme calcule toutes les distances les plus courtes entre toutes les 
                paires de sommets. Le diamètre correspond à la plus grande de ces distances.</p>
                
                <div class="complexity">
                    <strong>Complexité temporelle:</strong> O(V³) où V est le nombre de sommets<br>
                    <strong>Complexité spatiale:</strong> O(V²) pour la matrice des distances
                </div>
                
                <p><strong>Formule récursive:</strong><br>
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]) pour tout k</p>
            </div>
            
            <h3>2.3 Technologies Utilisées</h3>
            <ul>
                <li><strong>Python 3.x</strong> - Langage de programmation principal</li>
                <li><strong>NumPy</strong> - Calculs matriciels et opérations sur les tableaux</li>
                <li><strong>NetworkX</strong> - Bibliothèque de théorie des graphes</li>
                <li><strong>Matplotlib</strong> - Génération de graphiques et visualisations</li>
                <li><strong>Tkinter</strong> - Interface utilisateur graphique</li>
            </ul>
        </div>
        """
    
    def _generate_results(self, matrix, properties, distance_matrix, components):
        """Génère la section des résultats détaillés"""
        diameter_str = str(properties['diameter']) if properties['diameter'] != float('inf') else "∞"
        
        # Formatage de la matrice d'adjacence
        matrix_str = self._format_matrix_for_html(matrix)
        
        # Formatage de la matrice des distances
        distance_str = self._format_distance_matrix_for_html(distance_matrix)
        
        return f"""
        <div class="section">
            <h2>3. RÉSULTATS DÉTAILLÉS</h2>
            
            <h3>3.1 Caractéristiques du Graphe</h3>
            <table>
                <tr><th>Propriété</th><th>Valeur</th><th>Description</th></tr>
                <tr><td>Nombre de sommets</td><td>{properties['vertices']}</td><td>Ordre du graphe</td></tr>
                <tr><td>Nombre d'arêtes</td><td>{properties['edges']}</td><td>Taille du graphe</td></tr>
                <tr><td>Degré minimum</td><td>{properties['min_degree']}</td><td>Plus petit degré</td></tr>
                <tr><td>Degré maximum</td><td>{properties['max_degree']}</td><td>Plus grand degré</td></tr>
                <tr><td>Degré moyen</td><td>{properties['avg_degree']:.2f}</td><td>Moyenne des degrés</td></tr>
                <tr><td>Densité</td><td>{(2 * properties['edges']) / (properties['vertices'] * (properties['vertices'] - 1)):.3f}</td><td>Ratio arêtes/arêtes possibles</td></tr>
            </table>
            
            <h3>3.2 Matrice d'Adjacence</h3>
            <div class="matrix-container">
                <div class="matrix">{matrix_str}</div>
            </div>
            
            <h3>3.3 Analyse de Connexité</h3>
            <p>
                <strong>Résultat:</strong> Le graphe est <span class="highlight">{'CONNEXE' if properties['is_connected'] else 'NON CONNEXE'}</span>
            </p>
            
            {self._generate_components_analysis(components) if not properties['is_connected'] else ''}
            
            <h3>3.4 Calcul du Diamètre</h3>
            <p>
                <strong>Diamètre calculé:</strong> <span class="highlight">{diameter_str}</span>
            </p>
            
            <div class="matrix-container">
                <h4>Matrice des Distances (Floyd-Warshall)</h4>
                <div class="matrix">{distance_str}</div>
            </div>
            
            {'<p><strong>Interprétation:</strong> Le diamètre représente la plus grande distance géodésique entre deux sommets du graphe. Une valeur de ' + diameter_str + ' indique que certains sommets sont séparés par ' + diameter_str + ' arêtes au minimum.</p>' if properties['is_connected'] else '<p><strong>Interprétation:</strong> Le graphe étant non connexe, certaines paires de sommets ne sont pas reliées par un chemin, ce qui explique le diamètre infini.</p>'}
        </div>
        """
    
    def _generate_components_analysis(self, components):
        """Génère l'analyse des composantes connexes"""
        html = """
        <h4>Composantes Connexes Détectées</h4>
        <table>
            <tr><th>Composante</th><th>Sommets</th><th>Taille</th></tr>
        """
        
        for i, component in enumerate(components):
            sommets_str = ", ".join(map(str, sorted(component)))
            html += f"<tr><td>{i+1}</td><td>{sommets_str}</td><td>{len(component)}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_visualization(self, graph_image):
        """Génère la section de visualisation"""
        return f"""
        <div class="section page-break">
            <h2>4. VISUALISATION DU GRAPHE</h2>
            
            <div class="graph-image">
                <img src="data:image/png;base64,{graph_image}" alt="Visualisation du graphe" />
            </div>
            
            <p>
                La visualisation ci-dessus représente le graphe analysé. Les sommets sont colorés selon leur 
                appartenance aux composantes connexes :
            </p>
            <ul>
                <li><strong>Vert :</strong> Graphe connexe (tous les sommets dans une seule composante)</li>
                <li><strong>Différentes couleurs :</strong> Composantes connexes distinctes pour un graphe non connexe</li>
            </ul>
            
            <p>
                Les arêtes sont représentées par des liens entre les sommets, et la disposition utilise un 
                algorithme de forces pour optimiser la lisibilité du graphe. La taille des sommets peut 
                être proportionnelle à leur degré pour faciliter l'identification des nœuds importants.
            </p>
        </div>
        """
    
    def _generate_technical_details(self, matrix, properties):
        """Génère les détails techniques"""
        return f"""
        <div class="section">
            <h2>5. DÉTAILS TECHNIQUES ET IMPLÉMENTATION</h2>
            
            <h3>5.1 Structures de Données Utilisées</h3>
            <div class="algorithm-box">
                <p><strong>Matrice d'adjacence :</strong> Représentation sous forme de tableau NumPy {matrix.shape[0]}×{matrix.shape[1]}</p>
                <p><strong>Liste des sommets visités :</strong> Tableau booléen pour l'algorithme DFS</p>
                <p><strong>Matrice des distances :</strong> Matrice {matrix.shape[0]}×{matrix.shape[1]} pour Floyd-Warshall</p>
            </div>
            
            <h3>5.2 Optimisations Appliquées</h3>
            <ul>
                <li><strong>Vérification préalable :</strong> Test de la connexité avant le calcul du diamètre</li>
                <li><strong>Gestion des graphes orientés :</strong> Adaptation automatique des algorithmes</li>
                <li><strong>Optimisation mémoire :</strong> Utilisation de NumPy pour les calculs matriciels</li>
                <li><strong>Visualisation adaptative :</strong> Mise en page automatique selon la taille du graphe</li>
            </ul>
            
            <h3>5.3 Gestion des Cas Particuliers</h3>
            <div class="algorithm-box">
                <p><strong>Graphe vide :</strong> Diamètre = 0, connexité = False</p>
                <p><strong>Graphe à un sommet :</strong> Diamètre = 0, connexité = True</p>
                <p><strong>Graphe non connexe :</strong> Diamètre = ∞, analyse par composantes</p>
                <p><strong>Graphe complet :</strong> Diamètre = 1, densité maximale</p>
            </div>
            
            <h3>5.4 Validation des Résultats</h3>
            <p>
                Les résultats ont été validés par comparaison avec des calculs manuels sur des graphes 
                de petite taille et vérifiés à l'aide d'outils de référence comme NetworkX pour 
                s'assurer de la cohérence des algorithmes implémentés.
            </p>
        </div>
        """
    
    def _generate_conclusion(self, properties):
        """Génère la conclusion"""
        diameter_str = str(properties['diameter']) if properties['diameter'] != float('inf') else "infini"
        
        return f"""
        <div class="section">
            <h2>6. CONCLUSION ET PERSPECTIVES</h2>
            
            <h3>6.1 Synthèse des Résultats</h3>
            <p>
                L'analyse du graphe a révélé qu'il s'agit d'un graphe {'connexe' if properties['is_connected'] else 'non connexe'} 
                de {properties['vertices']} sommets et {properties['edges']} arêtes, avec un diamètre de {diameter_str}.
            </p>
            
            <p>
                {'Cette connexité garantit l\'existence d\'un chemin entre toute paire de sommets, ce qui est favorable pour de nombreuses applications pratiques comme les réseaux de communication ou les systèmes de transport.' if properties['is_connected'] else f'La non-connexité du graphe, avec ses {properties["components"]} composantes distinctes, indique une structure fragmentée qui pourrait nécessiter des stratégies d\'analyse spécifiques pour chaque composante.'}
            </p>
            
            <h3>6.2 Applications Pratiques</h3>
            <ul>
                <li><strong>Réseaux sociaux :</strong> Analyse de la cohésion et de la portée des communications</li>
                <li><strong>Transport :</strong> Optimisation des itinéraires et identification des goulots d'étranglement</li>
                <li><strong>Biologie :</strong> Étude des réseaux de protéines et des voies métaboliques</li>
                <li><strong>Internet :</strong> Analyse de la robustesse et de l'efficacité du réseau</li>
            </ul>
            
            <h3>6.3 Limites et Améliorations Possibles</h3>
            <p>
                L'algorithme Floyd-Warshall, bien qu'exact, présente une complexité O(V³) qui peut devenir 
                prohibitive pour de très grands graphes. Des alternatives comme l'algorithme de Johnson 
                ou des approximations pourraient être envisagées pour des graphes de plusieurs milliers de sommets.
            </p>
            
            <h3>6.4 Perspectives de Développement</h3>
            <ul>
                <li>Implémentation d'algorithmes plus efficaces pour les grands graphes</li>
                <li>Ajout d'analyses supplémentaires (centralité, clustering, etc.)</li>
                <li>Interface graphique plus avancée avec interactions en temps réel</li>
                <li>Support des graphes pondérés et orientés</li>
                <li>Export vers d'autres formats (LaTeX, PDF, JSON)</li>
            </ul>
        </div>
        """
    
    def _generate_appendix(self, matrix, distance_matrix):
        """Génère l'annexe avec les données complètes"""
        return f"""
        <div class="section page-break">
            <h2>7. ANNEXES</h2>
            
            <h3>7.1 Code Source des Algorithmes Principaux</h3>
            <div class="algorithm-box">
                <h4>Algorithme DFS pour la Connexité</h4>
                <pre style="font-size: 12px; background: #f8f9fa; padding: 10px; border-radius: 3px;">
def is_connected_dfs(matrix):
    n = len(matrix)
    if n == 0:
        return False
    
    visited = [False] * n
    
    def dfs(node):
        visited[node] = True
        for neighbor in range(n):
            if matrix[node][neighbor] and not visited[neighbor]:
                dfs(neighbor)
    
    dfs(0)
    return all(visited)
                </pre>
            </div>
            
            <div class="algorithm-box">
                <h4>Algorithme Floyd-Warshall pour les Distances</h4>
                <pre style="font-size: 12px; background: #f8f9fa; padding: 10px; border-radius: 3px;">
def floyd_warshall(matrix):
    n = len(matrix)
    dist = [[float('inf')] * n for _ in range(n)]
    
    # Initialisation
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            elif matrix[i][j]:
                dist[i][j] = 1
    
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist
                </pre>
            </div>
            
            <h3>7.2 Bibliographie et Références</h3>
            <ul>
                <li>Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). <em>Introduction to Algorithms</em>, 3rd Edition. MIT Press.</li>
                <li>Diestel, R. (2017). <em>Graph Theory</em>, 5th Edition. Springer Graduate Texts in Mathematics.</li>
                <li>Bondy, J. A., & Murty, U. S. R. (2008). <em>Graph Theory</em>. Springer Graduate Texts in Mathematics.</li>
                <li>NetworkX Documentation. (2024). <em>NetworkX: Network Analysis in Python</em>. https://networkx.org/</li>
            </ul>
            
            <h3>7.3 Glossaire</h3>
            <table>
                <tr><th>Terme</th><th>Définition</th></tr>
                <tr><td>Connexité</td><td>Propriété d'un graphe où il existe un chemin entre toute paire de sommets</td></tr>
                <tr><td>Diamètre</td><td>Plus grande distance géodésique entre deux sommets du graphe</td></tr>
                <tr><td>Distance géodésique</td><td>Longueur du plus court chemin entre deux sommets</td></tr>
                <tr><td>Composante connexe</td><td>Sous-graphe maximal où tous les sommets sont reliés</td></tr>
                <tr><td>Degré d'un sommet</td><td>Nombre d'arêtes incidentes à ce sommet</td></tr>
                <tr><td>Matrice d'adjacence</td><td>Représentation matricielle des connexions du graphe</td></tr>
                <tr><td>DFS</td><td>Depth-First Search - Parcours en profondeur</td></tr>
                <tr><td>Densité</td><td>Rapport entre le nombre d'arêtes et le nombre maximal possible</td></tr>
            </table>
        </div>
        """
    
    def _format_matrix_for_html(self, matrix):
        """
        Formate une matrice pour l'affichage HTML
        
        Args:
            matrix (numpy.ndarray): Matrice à formater
            
        Returns:
            str: Matrice formatée en texte
        """
        if matrix.size == 0:
            return "Matrice vide"
        
        lines = []
        n = matrix.shape[0]
        
        # En-tête avec numéros de colonnes
        header = "    " + "".join(f"{j:3d}" for j in range(n))
        lines.append(header)
        
        # Ligne de séparation
        separator = "   " + "-" * (3 * n + 1)
        lines.append(separator)
        
        # Lignes de la matrice
        for i in range(n):
            row = f"{i:2d} |"
            for j in range(n):
                row += f"{int(matrix[i][j]):3d}"
            lines.append(row)
        
        return "\n".join(lines)
    
    def _format_distance_matrix_for_html(self, distance_matrix):
        """
        Formate la matrice des distances pour l'affichage HTML
        
        Args:
            distance_matrix (list): Matrice des distances
            
        Returns:
            str: Matrice formatée en texte
        """
        if not distance_matrix or len(distance_matrix) == 0:
            return "Matrice vide"
        
        lines = []
        n = len(distance_matrix)
        
        # En-tête avec numéros de colonnes
        header = "    " + "".join(f"{j:4d}" for j in range(n))
        lines.append(header)
        
        # Ligne de séparation
        separator = "   " + "-" * (4 * n + 1)
        lines.append(separator)
        
        # Lignes de la matrice
        for i in range(n):
            row = f"{i:2d} |"
            for j in range(n):
                if distance_matrix[i][j] == float('inf'):
                    row += "  ∞ "
                else:
                    row += f"{int(distance_matrix[i][j]):4d}"
            lines.append(row)
        
        return "\n".join(lines)
    
    def _generate_footer(self):
        """Génère le pied de page"""
        return f"""
        <div class="footer">
            <p>
                <strong>Université de Yaoundé I - Département d'Informatique</strong><br>
                Rapport généré automatiquement le {datetime.datetime.now().strftime("%d/%m/%Y à %H:%M:%S")}<br>
                <em>Analyse de Graphes - Théorie et Applications</em>
            </p>
            
            <p style="margin-top: 10px; font-size: 10px;">
                Ce document a été généré par l'outil d'analyse de graphes développé dans le cadre du cours 
                de Théorie des Graphes. Pour toute question technique, veuillez contacter l'équipe de développement.
            </p>
        </div>
        """
    
    def generate_summary_stats(self, matrix):
        """
        Génère des statistiques résumées du graphe
        
        Args:
            matrix (numpy.ndarray): Matrice d'adjacence
            
        Returns:
            dict: Dictionnaire contenant les statistiques
        """
        n = matrix.shape[0]
        
        # Calcul des degrés
        if self.analyzer.is_directed(matrix):
            in_degrees = np.sum(matrix, axis=0)
            out_degrees = np.sum(matrix, axis=1)
            degrees = in_degrees + out_degrees
        else:
            degrees = np.sum(matrix, axis=1)
        
        # Calcul du nombre d'arêtes
        if self.analyzer.is_directed(matrix):
            edges = np.sum(matrix)
        else:
            edges = np.sum(matrix) // 2
        
        stats = {
            'vertices': n,
            'edges': int(edges),
            'min_degree': int(np.min(degrees)) if n > 0 else 0,
            'max_degree': int(np.max(degrees)) if n > 0 else 0,
            'avg_degree': float(np.mean(degrees)) if n > 0 else 0,
            'degree_sequence': sorted(degrees.tolist(), reverse=True),
            'density': float(2 * edges / (n * (n - 1))) if n > 1 else 0
        }
        
        return stats
    
    def export_to_json(self, graph_data, properties, filename=None):
        """
        Exporte les résultats en format JSON
        
        Args:
            graph_data (dict): Données du graphe
            properties (dict): Propriétés calculées
            filename (str): Nom du fichier (optionnel)
            
        Returns:
            str: Chemin du fichier JSON généré
        """
        import json
        
        # Préparation des données pour l'export
        export_data = {
            'metadata': {
                'generated_at': datetime.datetime.now().isoformat(),
                'generator': 'GraphAnalyzer Report Generator',
                'university': 'Université de Yaoundé I',
                'authors': [
                    'DIZE TCHEMOU MIGUEL CAREY',
                    'SAGUEN KAMDEM CHERYL RONALD',
                    'SIGNE FONGANG WILFRIED BRANDON'
                ]
            },
            'graph_properties': {
                'vertices': properties['vertices'],
                'edges': properties['edges'],
                'is_connected': properties['is_connected'],
                'diameter': properties['diameter'] if properties['diameter'] != float('inf') else None,
                'components': properties['components'],
                'is_directed': properties['is_directed'],
                'min_degree': properties['min_degree'],
                'max_degree': properties['max_degree'],
                'avg_degree': properties['avg_degree']
            },
            'adjacency_matrix': graph_data['data'].tolist() if hasattr(graph_data['data'], 'tolist') else graph_data['data']
        }
        
        # Génération du nom de fichier si non fourni
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analyse_graphe_{timestamp}.json"
        
        # Sauvegarde
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def generate_latex_report(self, graph_data, properties):
        """
        Génère un rapport au format LaTeX
        
        Args:
            graph_data (dict): Données du graphe
            properties (dict): Propriétés calculées
            
        Returns:
            str: Chemin du fichier LaTeX généré
        """
        matrix = graph_data['data']
        diameter_str = str(properties['diameter']) if properties['diameter'] != float('inf') else r"\infty"
        
        latex_content = rf"""
\documentclass[12pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[french]{{babel}}
\usepackage{{amsmath,amsfonts,amssymb}}
\usepackage{{graphicx}}
\usepackage{{geometry}}
\usepackage{{fancyhdr}}
\usepackage{{array}}
\usepackage{{booktabs}}

\geometry{{margin=2.5cm}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{Université de Yaoundé I}}
\fancyhead[R]{{Analyse de Graphe}}
\fancyfoot[C]{{\thepage}}

\title{{Rapport d'Analyse de Graphe:\\
Vérification de Connexité et Calcul du Diamètre}}
\author{{DIZE TCHEMOU MIGUEL CAREY \\ SAGUEN KAMDEM CHERYL RONALD \\ SIGNE FONGANG WILFRIED BRANDON}}
\date{{\today}}

\begin{{document}}

\maketitle

\section{{Résumé Exécutif}}

Ce rapport présente l'analyse complète d'un graphe comportant {properties['vertices']} sommets et {properties['edges']} arêtes. 

\begin{{center}}
\begin{{tabular}}{{|l|c|}}
\hline
\textbf{{Propriété}} & \textbf{{Valeur}} \\
\hline
État de Connexité & {'CONNEXE' if properties['is_connected'] else 'NON CONNEXE'} \\
Diamètre & ${diameter_str}$ \\
Nombre de Composantes & {properties['components']} \\
Type de Graphe & {'ORIENTÉ' if properties['is_directed'] else 'NON ORIENTÉ'} \\
\hline
\end{{tabular}}
\end{{center}}

\section{{Méthodologie}}

\subsection{{Vérification de la Connexité}}
L'algorithme de parcours en profondeur (DFS) est utilisé avec une complexité temporelle de $O(V + E)$.

\subsection{{Calcul du Diamètre}}
L'algorithme de Floyd-Warshall calcule toutes les distances avec une complexité de $O(V^3)$.

\section{{Résultats}}

Le graphe analysé {'est connexe' if properties['is_connected'] else 'n\'est pas connexe'} avec un diamètre de ${diameter_str}$.

\end{{document}}
        """
        
        # Sauvegarde
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rapport_graphe_{timestamp}.tex"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        return filename

# Fonction utilitaire pour l'utilisation du générateur
def create_graph_report(adjacency_matrix, output_format='html'):
    """
    Fonction utilitaire pour créer un rapport d'analyse de graphe
    
    Args:
        adjacency_matrix (list or numpy.ndarray): Matrice d'adjacence
        output_format (str): Format de sortie ('html', 'json', 'latex')
        
    Returns:
        str: Chemin du fichier généré
    """
    generator = ReportGenerator()
    
    # Préparation des données
    if not isinstance(adjacency_matrix, np.ndarray):
        adjacency_matrix = np.array(adjacency_matrix)
    
    graph_data = {
        'data': adjacency_matrix,
        'name': f'Graphe_{adjacency_matrix.shape[0]}x{adjacency_matrix.shape[1]}',
        'description': 'Graphe analysé automatiquement'
    }
    
    # Calcul des propriétés
    properties = generator.analyzer.get_graph_properties(adjacency_matrix)
    
    # Génération selon le format demandé
    if output_format.lower() == 'html':
        return generator.generate_report(graph_data)
    elif output_format.lower() == 'json':
        return generator.export_to_json(graph_data, properties)
    elif output_format.lower() == 'latex':
        return generator.generate_latex_report(graph_data, properties)
    else:
        raise ValueError(f"Format non supporté: {output_format}")
