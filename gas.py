# -*- coding: utf-8 -*-
"""Gas.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FyVPoj3273Ia31C2dOOzuco28xpVM3BL
"""
import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt

def gas_result():

    def find_last_name(ext: str) -> str:
      # Путь к папке с файлами (по умолчанию /content в Colab)
      directory = "/content"
    
      # Найти все файлы с расширением .rr
      rr_files = [f for f in os.listdir(directory) if f.endswith(ext)]
    
      if rr_files:
          # Найти последний загруженный файл по времени изменения
          latest_file = max(rr_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
          EXCEL_REPORT_PATH = os.path.join(directory, latest_file)
    
          print(f"Последний файл: {EXCEL_REPORT_PATH}")
      else:
          print(f"Файлы с расширением {ext} не найдены.")
      return EXCEL_REPORT_PATH
    
    EXCEL_REPORT_PATH = find_last_name(ext='rr')
    df1 = pd.read_csv(EXCEL_REPORT_PATH)
    df1.columns = ['Unnamed: 0']
    
    df_res = pd.DataFrame()
    df_res['ОВР'] = df1['Unnamed: 0'].dropna().reset_index(drop=True)
    df_res['ВРЕМЯ'] = df_res['ОВР'].cumsum()
    df_res = df_res.reset_index()
    
    # import re
    # match = re.search(r'([А-ЯЁ][а-яё]+\s[А-ЯЁ][а-яё]+\s[А-ЯЁ][а-яё]+)', EXCEL_REPORT_PATH)
    
    # if match:
    #     name = match.group(0)
    # else:
    #     full_name = EXCEL_REPORT_PATH.split('.')[0]
    
    name = EXCEL_REPORT_PATH.replace('/', ' ')
    
    GAS_REPORT_PATH = find_last_name(ext='xlsx')
    df = pd.read_excel(GAS_REPORT_PATH)
    
    #df['RER'] = df['VCO2'].iloc[2:] / df['VO2'].iloc[2:]
    df['RER'] = df['RQ']
    df['t'] = df['t'].apply(lambda x: str(x))
    df['VE'] = df['VE/VCO2'].iloc[2:] * df['VCO2'].iloc[2:]
    
    time_series = pd.Series(df_res['ОВР'])
    
    # 1. Кумулятивная сумма
    cumulative_time = time_series.cumsum()
    
    # 2. Определение целевых временных промежутков в миллисекундах
    target_times = pd.to_timedelta(df['t'].loc[2:].values).total_seconds() * 1000
    
    # 3. Поиск ближайших значений
    closest_indices = []
    for target in target_times:
        closest_index = (cumulative_time - target).abs().idxmin()  # Индекс ближайшего значения
        closest_indices.append(closest_index)
    
    # 4. Выбор элементов исходного временного ряда
    selected_elements = time_series.iloc[closest_indices]
    
    ser = selected_elements.reset_index(drop=True)
    empty_elements = pd.Series([None, None])  # Создаем Series с двумя пустыми значениями
    new_ovr = pd.concat([empty_elements, ser], ignore_index=True)
    
    df['RR'] = new_ovr
    
    from scipy.ndimage import gaussian_filter1d
    
    def replace_outliers_with_neighbors(data, threshold=1):
        """
        Заменяет выбросы на среднее значение соседей.
    
        :param data: Входные данные (список или массив).
        :param threshold: Порог для определения выбросов.
        :return: Массив с замененными выбросами.
        """
        mean = np.mean(data)
        std_dev = np.std(data)
    
        # Создаем маску для выбросов
        outlier_mask = np.abs(data - mean) > threshold * std_dev
    
        # Копируем исходные данные
        cleaned_data = data.copy()
    
        # Заменяем выбросы на среднее значение соседей
        for i in cleaned_data.index:
            if outlier_mask[i]:
                # Находим индексы соседей
                neighbors = []
                if i > 2:
                    neighbors.append(cleaned_data[i - 1])
                if i < cleaned_data.index[-1]:
                    neighbors.append(cleaned_data[i + 1])
    
                # Заменяем на среднее значение соседей, если они существуют
                if neighbors:
                    cleaned_data[i] = np.mean(neighbors)
    
        return cleaned_data
    
    
    def gaussian_smoothing(data, sigma):
        """
        Применяет гауссовское сглаживание к данным с сохранением длины.
    
        :param data: Входные данные (список или массив).
        :param sigma: Стандартное отклонение для гауссовского фильтра.
        :return: Сглаженные данные той же длины.
        """
        return gaussian_filter1d(data, sigma=sigma, mode='nearest')
    
    
    list_periods = ['VO2', 'VCO2', 'RER', 'VE', 'VE/VCO2', 'RR']
    for period in list_periods:
      data = replace_outliers_with_neighbors(df[period].iloc[2:].astype(float))
      arr = gaussian_smoothing(data, sigma=5)
      arr_with_nan = np.insert(arr, 0, [np.nan, np.nan])
      df[f'{period}_ga'] = arr_with_nan
    
    # def plot(df_res, name, ax, color):
    #     #groupby_list = list(range(0, len(df_res[name].loc[2:])))
    #     target_times = pd.to_timedelta(df['t'].loc[2:].values).total_seconds()
    
    #     sns.set(style="whitegrid")
    
    #     sns.lineplot(x=target_times, y=df_res[name].loc[2:], label=name, marker='o', color=color, ax=ax, markersize=4)
    
    #     ax.set_title(f'Динамика {name}')
    #     ax.set_xlabel('Время')
    #     ax.set_ylabel(name)
    #     ax.tick_params(axis='x', rotation=45)
    #     #ax.legend(title='Цвета кривых')
    #     ax.grid(True)
    
    
    # list_periods = ['VO2', 'VCO2', 'RER', 'VE', 'VE/VCO2', 'RR']
    # list_periods += [x+'_ga' for x in list_periods]
    # colors = ['blue', 'orange', 'green', 'red', 'purple', 'green']  # Список цветов
    # colors += colors
    # count_graphs = len(list_periods)
    # fig, axs = plt.subplots(nrows=count_graphs//2, ncols=2, figsize=(20, 12))  # 5 строк и 2 столбца
    # axs = axs.flatten()  # Преобразуем в одномерный массив для удобного доступа
    
    # # Перебираем все периоды и строим графики
    # for i, period in enumerate(list_periods):
    #     plot(df, period, axs[i], colors[i])  # Передаем соответствующий подграфик
    
    # # Удаляем пустые подграфики
    # if count_graphs % 2 != 0:
    #     fig.delaxes(axs[-1])  # Удаляем последний пустой подграфик
    # plt.tight_layout()
    # plt.show()
    
    import pandas as pd
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    
    def plot_single(fig, row, col, df, period, color):
        # Преобразуем время в секунды
        target_times = pd.to_timedelta(df['t'].loc[2:]).dt.total_seconds()
    
        # Добавляем линию с точками на соответствующий подграфик
        fig.add_trace(go.Scatter(
            x=target_times,
            y=df[period].loc[2:],  # Предполагается, что df содержит нужные данные
            mode='lines+markers',
            name=period,
            marker=dict(size=4, color=color),
            text=[f'Время: {time:.0f} с, {period}: {value:.2f}, RR: {rr:.0f} мс' for time, value, rr in zip(target_times, df[period].loc[2:], df['RR'].loc[2:])],
            hoverinfo='text'
        ), row=row, col=col)
    
    # Предполагаем, что df уже загружен и содержит необходимые данные
    list_periods = ['VO2', 'VCO2', 'RER', 'VE', 'VE/VCO2', 'RR']
    list_periods += [x+'_ga' for x in list_periods]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'green']  # Список цветов
    colors += colors
    count_graphs = len(list_periods)
    
    # Создаем фигуру с подграфиками (2 столбца)
    fig = make_subplots(rows=(count_graphs + 1) // 2, cols=2,
                        subplot_titles=list_periods)
    
    # Перебираем все периоды и строим графики
    for i, period in enumerate(list_periods):
        row = i // 2 + 1  # Номер строки
        col = i % 2 + 1   # Номер столбца
    
        # Вызываем функцию для построения графика в одной ячейке
        plot_single(fig, row, col, df, period, colors[i])
        fig.update_xaxes(title_text='Время (сек)', row=row, col=col)
    
    # Настраиваем внешний вид графика
    fig.update_layout(
        title=f'Динамика показателей газового анализа {name}',
        height=600 + (count_graphs // 2) * 150,  # Высота графика в зависимости от количества подграфиков
        showlegend=False,
    )
    
    
    # Отображаем график
    fig.show()
    
    html_path = f"gas_{name}.html"
    fig.write_html(html_path)
    
    # Добавляем кастомный JavaScript для копирования координат
    custom_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var plot = document.getElementsByClassName('js-plotly-plot')[0];
    
        // Отключаем контекстное меню по правому клику
        plot.oncontextmenu = function(event) {
            event.preventDefault();
        };
    
        // Обработчик клика на точке
        plot.on('plotly_click', function(data) {
            if (data.points.length > 0) {
                var point = data.points[0];
                var hoverText = point.text; // Берем текст из hover
    
                // Копируем в буфер обмена
                navigator.clipboard.writeText(hoverText);
            }
        });
    });
    </script>
    """
    
    # Экспортируем HTML с кастомным скриптом
    
    with open(html_path, 'a') as f:
        f.write(custom_js)
    
    from google.colab import files
    files.download(html_path)
