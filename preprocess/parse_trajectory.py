import json
from bs4 import BeautifulSoup

def parse_single_step(children):
    step = {}
    for child in children:
        if child.get('class') == ['ts-header']:
            id = child.find('h2').get_text()
            step['Step'] = int(id.split(' ')[-1])
            step['url'] = child.find('a').get('href')
        elif child.get('class') == ['predict_action']:
            elements = child.find_all('pre')
            step['Predict Action'] = elements[0].get_text()
            step['Parsed Action'] = elements[2].get_text()
        elif child.get_text() == 'Image Observation':
            img = child.find_next_sibling().find('img')
            if img:
                step['Image Observation'] = img['src']
        elif child.get_text().startswith('Grounding @'):
            img = child.find_next_sibling().find('img')
            if img:
                step['Page Grounding'] = img['src']
        elif 'page_screenshot' in child.get('class'):
            img = child.find('img')
            if img:
                step['Page Screenshot'] = img['src']
        else:
            button = child.find('button')
            if button:
                name = button.get_text()
                if name.lower() in ['text observation','reasoning history', 'action history', 'target element bbox']:
                    pre = child.find('pre')
                    step[name] = pre.get_text()

    return step

def parse_trajectory(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    soup = BeautifulSoup(content, 'html5lib')

    preamble = soup.find('pre').get_text(strip=True)
    task_id = preamble.split('task_id: ')[1].split("\n")[0]
    intent = preamble.split('intent: ')[1].split("\n")[0]
    start_url = preamble.split('start_url: ')[1].split("\n")[0]
    license_task_metadata = None
    if 'license_task_metadata: ' in preamble:
        try:
            license_task_metadata = eval(preamble.split('license_task_metadata: ')[1].split('\n')[0])
        except Exception as e:
            pass

    run_args = json.loads(soup.find('div', class_='run_args').find('pre').get_text(strip=True))

    # Parse the trajectory
    step_data = []
    body = soup.find('body')
    body_children = body.find_all(recursive=False)
    for child in body_children:
        if child.get('class') == ['ts-wrapper']:
            # Get all direct children
            children = child.find_all(recursive=False)
            step = parse_single_step(children)
            step_data.append(step)

    return {
        'task_id': task_id,
        'intent': intent,
        'run_args': run_args,
        'start_url': start_url,
        'trajectory': step_data,
        'license_task_metadata': license_task_metadata,
    }
