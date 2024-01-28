from setuptools import setup, find_packages

setup(
    name='ragxplorer',
    version='0.1.8',
    author='Gabriel Chua',
    author_email='cyzgab@gmail.com',
    description='A open-source tool to to visualise your RAG documents 🔮.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/gabrielchua/ragxplorer',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'umap-learn',
        'sentence-transformers',
        'plotly',
        'tqdm',
        'PyPDF2',
        'langchain',
        'chromadb',
        'openai',
        'pydantic'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
