import efinance as ef
import pandas as pd

stock_df = pd.read_csv('SSE50.csv')

stock_code_list = stock_df['股票代码']
print(stock_code_list)

beg = '20220101'
end = '20220630'

for stock_code in stock_code_list:
    df = ef.stock.get_quote_history(str(stock_code), fqt=1)
    df.to_csv(f'{stock_code}.csv', encoding='utf-8-sig', index=None)
# print(df.head())