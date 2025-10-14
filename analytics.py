import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import json

class ResumeAnalyticsDashboard:
    def __init__(self, resume_data_path='enhanced_resume_data.csv'):
        """Initialize the dashboard"""
        self.resume_df = pd.read_csv(resume_data_path)
        self.setup_style()
    
    def setup_style(self):
        """Setup plotting style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def analyze_skills_distribution(self, save_path='skills_analysis.png'):
        """Analyze and visualize skills distribution"""
        # Parse skills from string representation
        all_skills = []
        for skills in self.resume_df['skills']:
            if pd.notna(skills) and skills:
                try:
                    skill_list = eval(skills) if isinstance(skills, str) else skills
                    all_skills.extend(skill_list)
                except:
                    pass
        
        # Count skills
        skill_counts = Counter(all_skills)
        top_skills = dict(skill_counts.most_common(20))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        skills = list(top_skills.keys())
        counts = list(top_skills.values())
        
        ax1.barh(skills, counts, color='steelblue')
        ax1.set_xlabel('Number of Resumes', fontsize=12, fontweight='bold')
        ax1.set_title('Top 20 Skills in Resume Dataset', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # Word cloud
        wordcloud = WordCloud(width=800, height=600, 
                             background_color='white',
                             colormap='viridis').generate_from_frequencies(skill_counts)
        
        ax2.imshow(wordcloud, interpolation='bilinear')
        ax2.axis('off')
        ax2.set_title('Skills Word Cloud', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Skills analysis saved to {save_path}")
        return skill_counts
    
    def analyze_category_distribution(self, save_path='category_distribution.png'):
        """Analyze resume categories"""
        category_counts = self.resume_df['category'].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        category_counts.plot(kind='bar', ax=ax1, color='coral')
        ax1.set_title('Resume Distribution by Category', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Resumes', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        ax2.pie(category_counts.values, labels=category_counts.index, 
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('Category Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Category analysis saved to {save_path}")
        return category_counts
    
    def analyze_experience_levels(self, save_path='experience_analysis.png'):
        """Analyze experience levels"""
        exp_data = self.resume_df['experience_years'].dropna()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        ax1.hist(exp_data, bins=15, color='mediumseagreen', edgecolor='black', alpha=0.7)
        ax1.set_title('Distribution of Experience Years', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Years of Experience', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Resumes', fontsize=12, fontweight='bold')
        ax1.axvline(exp_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {exp_data.mean():.1f} years')
        ax1.legend()
        
        # Box plot
        ax2.boxplot(exp_data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='navy'),
                   medianprops=dict(color='red', linewidth=2),
                   whiskerprops=dict(color='navy'),
                   capprops=dict(color='navy'))
        ax2.set_title('Experience Distribution (Box Plot)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Years of Experience', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Experience analysis saved to {save_path}")
        print(f"Average Experience: {exp_data.mean():.2f} years")
        print(f"Median Experience: {exp_data.median():.2f} years")
        return exp_data.describe()
    
    def analyze_text_metrics(self, save_path='text_metrics.png'):
        """Analyze resume text metrics"""
        # Parse text metrics
        word_counts = []
        for metrics in self.resume_df['text_metrics']:
            if pd.notna(metrics):
                try:
                    metric_dict = eval(metrics) if isinstance(metrics, str) else metrics
                    word_counts.append(metric_dict.get('word_count', 0))
                except:
                    pass
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Word count distribution
        ax1.hist(word_counts, bins=30, color='plum', edgecolor='black', alpha=0.7)
        ax1.set_title('Resume Word Count Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Word Count', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Resumes', fontsize=12, fontweight='bold')
        ax1.axvline(np.mean(word_counts), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(word_counts):.0f} words')
        ax1.legend()
        
        # Scatter: Skills count vs Word count
        skills_counts = []
        for skills in self.resume_df['skills']:
            if pd.notna(skills):
                try:
                    skill_list = eval(skills) if isinstance(skills, str) else skills
                    skills_counts.append(len(skill_list))
                except:
                    skills_counts.append(0)
            else:
                skills_counts.append(0)
        
        # Match lengths
        min_len = min(len(word_counts), len(skills_counts))
        ax2.scatter(word_counts[:min_len], skills_counts[:min_len], 
                   alpha=0.5, color='teal', s=50)
        ax2.set_title('Skills Count vs Resume Length', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Word Count', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Skills', fontsize=12, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(word_counts[:min_len], skills_counts[:min_len], 1)
        p = np.poly1d(z)
        ax2.plot(word_counts[:min_len], p(word_counts[:min_len]), 
                "r--", linewidth=2, label='Trend')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Text metrics analysis saved to {save_path}")
        return word_counts
    
    def analyze_skills_by_category(self, save_path='skills_by_category.png'):
        """Analyze skills distribution across categories"""
        category_skills = {}
        
        for idx, row in self.resume_df.iterrows():
            category = row['category']
            if pd.notna(row['skills']):
                try:
                    skills = eval(row['skills']) if isinstance(row['skills'], str) else row['skills']
                    if category not in category_skills:
                        category_skills[category] = []
                    category_skills[category].extend(skills)
                except:
                    pass
        
        # Get top 5 skills per category
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (category, skills) in enumerate(list(category_skills.items())[:6]):
            skill_counts = Counter(skills)
            top_skills = dict(skill_counts.most_common(10))
            
            if top_skills:
                axes[idx].barh(list(top_skills.keys()), list(top_skills.values()), 
                              color=plt.cm.Set3(idx))
                axes[idx].set_title(f'{category}', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel('Frequency')
                axes[idx].invert_yaxis()
        
        # Hide extra subplots
        for idx in range(len(category_skills), 6):
            axes[idx].axis('off')
        
        plt.suptitle('Top Skills by Category', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Skills by category analysis saved to {save_path}")
        return category_skills
    
    def create_comprehensive_report(self):
        """Generate a comprehensive analytics report"""
        print("=" * 80)
        print("RESUME DATASET ANALYTICS REPORT")
        print("=" * 80)
        
        print(f"\nTotal Resumes: {len(self.resume_df)}")
        print(f"Total Categories: {self.resume_df['category'].nunique()}")
        
        print("\n" + "=" * 80)
        print("1. CATEGORY DISTRIBUTION")
        print("=" * 80)
        category_dist = self.analyze_category_distribution()
        
        print("\n" + "=" * 80)
        print("2. SKILLS ANALYSIS")
        print("=" * 80)
        skills_dist = self.analyze_skills_distribution()
        print(f"Total Unique Skills: {len(skills_dist)}")
        print(f"Top 5 Skills: {list(dict(skills_dist.most_common(5)).keys())}")
        
        print("\n" + "=" * 80)
        print("3. EXPERIENCE ANALYSIS")
        print("=" * 80)
        exp_stats = self.analyze_experience_levels()
        
        print("\n" + "=" * 80)
        print("4. TEXT METRICS ANALYSIS")
        print("=" * 80)
        self.analyze_text_metrics()
        
        print("\n" + "=" * 80)
        print("5. SKILLS BY CATEGORY")
        print("=" * 80)
        self.analyze_skills_by_category()
        
        print("\n" + "=" * 80)
        print("REPORT GENERATION COMPLETE")
        print("=" * 80)
        
        # Save summary to JSON
        summary = {
            'total_resumes': len(self.resume_df),
            'total_categories': int(self.resume_df['category'].nunique()),
            'category_distribution': category_dist.to_dict(),
            'top_skills': dict(skills_dist.most_common(20)),
            'experience_stats': {
                'mean': float(self.resume_df['experience_years'].mean()),
                'median': float(self.resume_df['experience_years'].median()),
                'min': float(self.resume_df['experience_years'].min()),
                'max': float(self.resume_df['experience_years'].max())
            }
        }
        
        with open('analytics_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\nSummary saved to 'analytics_summary.json'")


# Usage
if __name__ == "__main__":
    dashboard = ResumeAnalyticsDashboard()
    dashboard.create_comprehensive_report()