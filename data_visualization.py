import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from pathlib import Path

# Load the uploaded CSV file
current_folder = Path(__file__).parent.resolve()
root_folder = current_folder.parent.parent
output_folder = str(Path(root_folder, "output"))
data_folder = str(Path(current_folder, "outputs"))
filename = "weekly_posts_narrative_tone_ltm3_attention_sentiment.csv"
file_path = os.path.join(data_folder, filename)
data = pd.read_csv(file_path)

# Convert 'week' column to datetime format
data['week'] = pd.to_datetime(data['week'])

# Define LTM 3 factors (bubble_TF, scam_TF, intrinsic_value_TF) and market size/momentum factors (cmkt, csize, cmom)
ltm3_factors = ['bubble_TF', 'scam_TF', 'intrinsic_value_TF']
market_factors = ['cmkt', 'csize', 'cmom']
narrative_factors = ['bubble_TF', 'scam_TF', 'intrinsic_value_TF', 'inflation_TF', 
                     'volatility_TF', 'security_TF', 'regulation_TF', 'blockchain_technology_TF',
                     'fear_of_missing_out_TF', 'fear_of_loss_TF', 'environmental_TF', 
                     'trading_strategy_TF', 'liquidity_TF']

# Updated function to avoid color conflicts and make it more professional
def plot_with_two_y_axes(ax1, x_data, y1_data, y2_data, y1_label, y2_label, y1_color, y2_color):
    ax1.set_xlabel('Week')
    ax1.set_ylabel(y1_label, color=y1_color)
    ax1.plot(x_data, y1_data, color=y1_color, label=y1_label)
    ax1.tick_params(axis='y', labelcolor=y1_color)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel(y2_label, color=y2_color)
    ax2.plot(x_data, y2_data, color=y2_color, label=y2_label)
    ax2.tick_params(axis='y', labelcolor=y2_color)
    return ax1, ax2


# Define the plot function for dual y-axes
def plot_with_two_y_axes(ax1, x_data, y1_data, y2_data, y1_label, y2_label, y1_color, y2_color):
    ax1.set_xlabel('Week')
    ax1.set_ylabel(y1_label, color=y1_color)
    ax1.plot(x_data, y1_data, color=y1_color, label=y1_label)
    ax1.tick_params(axis='y', labelcolor=y1_color)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel(y2_label, color=y2_color)
    ax2.plot(x_data, y2_data, color=y2_color, label=y2_label)
    ax2.tick_params(axis='y', labelcolor=y2_color)
    return ax1, ax2

# Plot 1: Bitcoin return and market factors (cmkt, csize, cmom)
def plot_first_graph():
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1, ax2 = plot_with_two_y_axes(ax1, data['week'], data['bitret'], data['cmkt'], 'Bitcoin Return', 'Market Size (cmkt)', 'tab:blue', 'tab:orange')
    ax1.plot(data['week'], data['csize'], label='Coin Size (csize)', color='tab:green', linestyle='--')
    ax1.plot(data['week'], data['cmom'], label='Momentum (cmom)', color='tab:red', linestyle='-.')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=12))

    plt.title('Bitcoin Return, Market Size (cmkt), Coin Size (csize), and Momentum (cmom)')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize='small')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')

    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save as PDF
    pdf_file_path = os.path.join(output_folder, 'bitcoin_ltm3.pdf')
    plt.savefig(pdf_file_path)
    plt.show()

    return pdf_file_path

# Plot 2: Bitcoin return and Attention
def plot_second_graph():
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1, ax2 = plot_with_two_y_axes(ax1, data['week'], data['bitret'], data['attention'], 'Bitcoin Return', 'Attention', 'tab:blue', 'tab:orange')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=12))

    plt.title('Bitcoin Return and Attention Over Time')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize='small')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')

    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save as PDF
    pdf_file_path = os.path.join(output_folder, 'bitcoin_attention.pdf')
    plt.savefig(pdf_file_path)
    plt.show()

    return pdf_file_path

# Plot 3: Bitcoin return, Attention, Sentiment LM, and Bubble Narrative
def plot_third_graph():
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1, ax2 = plot_with_two_y_axes(ax1, data['week'], data['bitret'], data['attention'], 'Bitcoin Return', 'Attention', 'tab:blue', 'tab:orange')

    ax1.plot(data['week'], data['sentiment_LM'], label='Sentiment LM', color='tab:purple', linestyle='-.')
    ax1.plot(data['week'], data['bubble_TF'], label='Bubble Narrative', color='tab:green', linestyle=':')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=12))

    plt.title('Bitcoin Return, Attention, Sentiment LM, and Bubble Narrative Over Time')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize='small')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')

    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save as PDF
    pdf_file_path = os.path.join(output_folder, 'bitcoin_attention_sentiment_bubble_narratives.pdf')
    plt.savefig(pdf_file_path)
    plt.show()

def plot_fourth_graph():
    # plot the narratives factors and bitcoin return
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1, ax2 = plot_with_two_y_axes(ax1, data['week'], data['bitret'], data['bubble_TF'], 'Bitcoin Return', 'Bubble Narrative', 'tab:blue', 'tab:orange')

    ax1.plot(data['week'], data['scam_TF'], label='Scam Narrative', color='tab:green', linestyle='--')
    ax1.plot(data['week'], data['intrinsic_value_TF'], label='Intrinsic Value Narrative', color='tab:red', linestyle='-.')
    ax1.plot(data['week'], data['inflation_TF'], label='Inflation Narrative', color='tab:purple', linestyle=':')
    ax1.plot(data['week'], data['volatility_TF'], label='Volatility Narrative', color='tab:orange', linestyle='-')
    ax1.plot(data['week'], data['security_TF'], label='Security Narrative', color='tab:brown', linestyle='--')
    ax1.plot(data['week'], data['regulation_TF'], label='Regulation Narrative', color='tab:pink', linestyle='-.')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=12))

    plt.title('Bitcoin Return and Narrative Factors Over Time')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize='small')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')

    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save as PDF
    pdf_file_path = os.path.join(output_folder, 'bitcoin_narratives.pdf')
    plt.savefig(pdf_file_path)
    plt.show()

if __name__ == "__main__":
    plot_first_graph()
    plot_second_graph()
    plot_third_graph()
    plot_fourth_graph()
