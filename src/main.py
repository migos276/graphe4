#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyseur de Graphes - Module Principal
Université de Yaoundé I - Département d'Informatique

Auteurs: DIZE TCHEMOU MIGUEL CAREY, SAGUEN KAMDEM CHERYL RONALD, 
         SIGNE FONGANG WILFRIED BRANDON
Supervisé par: Dr Manga MAXWELL
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from graph_core import GraphAnalyzer
from report_generator import ReportGenerator
import json

class GraphAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Analyseur de Graphes - Connexité et Diamètre")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        self.analyzer = GraphAnalyzer()
        self.report_gen = ReportGenerator()
        self.current_graph = None
        
        self.setup_ui()
        self.show_guide()
    
    def setup_ui(self):
        """Initialise l'interface utilisateur"""
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Titre
        title_label = tk.Label(main_frame, text="Analyseur de Connexité et Diamètre de Graphes", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=(0, 20))
        
        # Frame pour les boutons de navigation
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(nav_frame, text="📖 Guide d'utilisation", 
                  command=self.show_guide).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(nav_frame, text="📄 Générer Rapport", 
                  command=self.generate_report).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(nav_frame, text="💾 Sauvegarder Graphe", 
                  command=self.save_graph).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(nav_frame, text="📂 Charger Graphe", 
                  command=self.load_graph).pack(side=tk.LEFT)
        
        # Notebook pour les onglets
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Onglet Matrice d'Adjacence
        self.setup_matrix_tab()
        
        # Onglet Liste d'Adjacence
        self.setup_list_tab()
        
        # Onglet Visualisation
        self.setup_visualization_tab()
        
        # Onglet Résultats
        self.setup_results_tab()
    
    def setup_matrix_tab(self):
        """Configuration de l'onglet matrice d'adjacence"""
        matrix_frame = ttk.Frame(self.notebook)
        self.notebook.add(matrix_frame, text="Matrice d'Adjacence")
        
        # Instructions
        inst_label = tk.Label(matrix_frame, 
                            text="Entrez la matrice d'adjacence (séparée par des espaces, une ligne par rangée):",
                            font=('Arial', 10), wraplength=400)
        inst_label.pack(pady=10)
        
        # Zone de texte pour la matrice
        self.matrix_text = tk.Text(matrix_frame, height=10, width=50, font=('Courier', 10))
        self.matrix_text.pack(pady=10)
        
        # Exemple par défaut
        default_matrix = "0 1 0 1\n1 0 1 0\n0 1 0 1\n1 0 1 0"
        self.matrix_text.insert('1.0', default_matrix)
        
        # Bouton d'analyse
        ttk.Button(matrix_frame, text="🔍 Analyser Matrice", 
                  command=self.analyze_matrix).pack(pady=10)
    
    def setup_list_tab(self):
        """Configuration de l'onglet liste d'adjacence"""
        list_frame = ttk.Frame(self.notebook)
        self.notebook.add(list_frame, text="Liste d'Adjacence")
        
        # Instructions
        inst_label = tk.Label(list_frame,
                            text="Entrez la liste d'adjacence (format: sommet: voisin1,voisin2,...):",
                            font=('Arial', 10), wraplength=400)
        inst_label.pack(pady=10)
        
        # Zone de texte pour la liste
        self.list_text = tk.Text(list_frame, height=10, width=50, font=('Courier', 10))
        self.list_text.pack(pady=10)
        
        # Exemple par défaut
        default_list = "0: 1,3\n1: 0,2\n2: 1,3\n3: 0,2"
        self.list_text.insert('1.0', default_list)
        
        # Bouton d'analyse
        ttk.Button(list_frame, text="🔍 Analyser Liste", 
                  command=self.analyze_list).pack(pady=10)
    
    def setup_visualization_tab(self):
        """Configuration de l'onglet visualisation"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="Visualisation")
        
        # Canvas pour matplotlib
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Frame pour les contrôles
        controls_frame = ttk.Frame(viz_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(controls_frame, text="🎨 Redessiner", 
                  command=self.redraw_graph).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="💾 Sauvegarder Image", 
                  command=self.save_graph_image).pack(side=tk.LEFT, padx=5)
    
    def setup_results_tab(self):
        """Configuration de l'onglet résultats"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Résultats")
        
        # Zone de texte pour les résultats
        self.results_text = tk.Text(results_frame, height=20, width=80, font=('Courier', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.results_text.configure(yscrollcommand=scrollbar.set)
    
    def analyze_matrix(self):
        """Analyse le graphe à partir de la matrice d'adjacence"""
        try:
            matrix_text = self.matrix_text.get('1.0', tk.END).strip()
            matrix = self.analyzer.parse_adjacency_matrix(matrix_text)
            
            # Analyse
            is_connected = self.analyzer.is_connected(matrix)
            diameter = self.analyzer.calculate_diameter(matrix)
            
            # Stockage pour le rapport
            self.current_graph = {
                'type': 'matrix',
                'data': matrix,
                'connected': is_connected,
                'diameter': diameter,
                'nodes': len(matrix)
            }
            
            # Affichage des résultats
            self.display_results(matrix, is_connected, diameter)
            
            # Visualisation
            self.analyzer.visualize_graph(matrix, self.ax)
            self.canvas.draw()
            
            # Basculer vers l'onglet résultats
            self.notebook.select(3)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse: {str(e)}")
    
    def analyze_list(self):
        """Analyse le graphe à partir de la liste d'adjacence"""
        try:
            list_text = self.list_text.get('1.0', tk.END).strip()
            adj_dict = self.analyzer.parse_adjacency_list(list_text)
            matrix = self.analyzer.adjacency_list_to_matrix(adj_dict)
            
            # Analyse
            is_connected = self.analyzer.is_connected(matrix)
            diameter = self.analyzer.calculate_diameter(matrix)
            
            # Stockage pour le rapport
            self.current_graph = {
                'type': 'list',
                'data': matrix,
                'adj_list': adj_dict,
                'connected': is_connected,
                'diameter': diameter,
                'nodes': len(matrix)
            }
            
            # Affichage des résultats
            self.display_results(matrix, is_connected, diameter)
            
            # Visualisation
            self.analyzer.visualize_graph(matrix, self.ax)
            self.canvas.draw()
            
            # Basculer vers l'onglet résultats
            self.notebook.select(3)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse: {str(e)}")
    
    def display_results(self, matrix, is_connected, diameter):
        """Affiche les résultats de l'analyse"""
        self.results_text.delete('1.0', tk.END)
        
        results = f"""
═══════════════════════════════════════════════════════════════
                    RÉSULTATS DE L'ANALYSE DU GRAPHE
═══════════════════════════════════════════════════════════════

📊 INFORMATIONS GÉNÉRALES:
   • Nombre de sommets: {len(matrix)}
   • Nombre d'arêtes: {np.sum(matrix) // 2}
   • Type de graphe: {"Orienté" if not self.analyzer.is_symmetric(matrix) else "Non-orienté"}

🔗 ANALYSE DE CONNEXITÉ:
   • Le graphe est {"CONNEXE" if is_connected else "NON CONNEXE"}
   
🎯 CALCUL DU DIAMÈTRE:
   • Diamètre: {diameter if diameter != float('inf') else "∞ (graphe non connexe)"}
   
📋 MATRICE D'ADJACENCE:
{self.format_matrix(matrix)}

📈 MATRICE DES DISTANCES:
{self.format_matrix(self.analyzer.floyd_warshall(matrix))}

🔍 DÉTAILS DE L'ALGORITHME:
   • Algorithme utilisé pour la connexité: Parcours en profondeur (DFS)
   • Algorithme utilisé pour le diamètre: Floyd-Warshall
   • Complexité temporelle: O(n³) pour Floyd-Warshall
   • Complexité spatiale: O(n²)

💡 INTERPRÉTATION:
   {"Le graphe est connexe, ce qui signifie qu'il existe un chemin entre toute paire de sommets." if is_connected else "Le graphe n'est pas connexe, il contient plusieurs composantes connexes."}
   {"Le diamètre représente la plus grande distance entre deux sommets du graphe." if is_connected else ""}
        """
        
        self.results_text.insert('1.0', results)
    
    def format_matrix(self, matrix):
        """Formate une matrice pour l'affichage"""
        if isinstance(matrix[0][0], float) and matrix[0][0] == float('inf'):
            # Matrice des distances avec infini
            formatted = []
            for row in matrix:
                formatted_row = []
                for val in row:
                    if val == float('inf'):
                        formatted_row.append("∞")
                    else:
                        formatted_row.append(str(int(val)))
                formatted.append("   ".join(f"{val:>3}" for val in formatted_row))
            return "\n".join(formatted)
        else:
            # Matrice normale
            return "\n".join("   ".join(f"{val:>3}" for val in row) for row in matrix)
    
    def generate_report(self):
        """Génère le rapport HTML"""
        if not self.current_graph:
            messagebox.showwarning("Attention", "Veuillez d'abord analyser un graphe!")
            return
        
        try:
            report_path = self.report_gen.generate_report(self.current_graph)
            messagebox.showinfo("Succès", f"Rapport généré: {report_path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la génération: {str(e)}")
    
    def save_graph(self):
        """Sauvegarde le graphe actuel"""
        if not self.current_graph:
            messagebox.showwarning("Attention", "Aucun graphe à sauvegarder!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Conversion des numpy arrays en listes pour JSON
                graph_data = self.current_graph.copy()
                graph_data['data'] = [row.tolist() if hasattr(row, 'tolist') else row 
                                     for row in graph_data['data']]
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("Succès", f"Graphe sauvegardé: {filename}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {str(e)}")
    
    def load_graph(self):
        """Charge un graphe sauvegardé"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.current_graph = json.load(f)
                
                # Reconversion en numpy array
                matrix = np.array(self.current_graph['data'])
                self.current_graph['data'] = matrix
                
                # Affichage
                self.display_results(matrix, self.current_graph['connected'], 
                                   self.current_graph['diameter'])
                self.analyzer.visualize_graph(matrix, self.ax)
                self.canvas.draw()
                
                messagebox.showinfo("Succès", f"Graphe chargé: {filename}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors du chargement: {str(e)}")
    
    def redraw_graph(self):
        """Redessine le graphe"""
        if self.current_graph:
            self.analyzer.visualize_graph(self.current_graph['data'], self.ax)
            self.canvas.draw()
    
    def save_graph_image(self):
        """Sauvegarde l'image du graphe"""
        if not self.current_graph:
            messagebox.showwarning("Attention", "Aucun graphe à sauvegarder!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Succès", f"Image sauvegardée: {filename}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {str(e)}")
    
    def show_guide(self):
        """Affiche le guide d'utilisation"""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Guide d'utilisation")
        guide_window.geometry("600x500")
        guide_window.configure(bg='white')
        
        guide_text = tk.Text(guide_window, wrap=tk.WORD, font=('Arial', 10), bg='white')
        guide_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        guide_content = """
📖 GUIDE D'UTILISATION - ANALYSEUR DE GRAPHES

🎯 OBJECTIF:
Ce programme analyse la connexité d'un graphe et calcule son diamètre.

📝 FORMATS D'ENTRÉE:

1️⃣ MATRICE D'ADJACENCE:
   • Entrez chaque ligne de la matrice sur une ligne séparée
   • Séparez les valeurs par des espaces
   • Exemple pour un graphe à 4 sommets:
     0 1 0 1
     1 0 1 0
     0 1 0 1
     1 0 1 0

2️⃣ LISTE D'ADJACENCE:
   • Format: sommet: voisin1,voisin2,...
   • Un sommet par ligne
   • Exemple:
     0: 1,3
     1: 0,2
     2: 1,3
     3: 0,2

🔧 FONCTIONNALITÉS:

✅ Analyse de connexité (algorithme DFS)
✅ Calcul du diamètre (algorithme Floyd-Warshall)
✅ Visualisation graphique du graphe
✅ Génération de rapport académique HTML
✅ Sauvegarde/Chargement de graphes
✅ Export d'images

📊 RÉSULTATS FOURNIS:

• Connexité du graphe (connexe/non connexe)
• Diamètre du graphe
• Matrice des distances
• Visualisation du graphe
• Statistiques détaillées

🚀 UTILISATION:

1. Choisissez un onglet (Matrice ou Liste d'Adjacence)
2. Entrez vos données de graphe
3. Cliquez sur "Analyser"
4. Consultez les résultats dans l'onglet "Résultats"
5. Générez un rapport avec le bouton "Générer Rapport"

💡 CONSEILS:
• Utilisez des graphes non orientés pour de meilleurs résultats
• Les boucles (arêtes d'un sommet vers lui-même) sont supportées
• Pour les graphes non connexes, le diamètre sera infini
        """
        
        guide_text.insert('1.0', guide_content)
        guide_text.config(state=tk.DISABLED)
        
        ttk.Button(guide_window, text="Fermer", 
                  command=guide_window.destroy).pack(pady=10)

def main():
    """Fonction principale"""
    root = tk.Tk()
    app = GraphAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
