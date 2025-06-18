# portfolio.llm
web3.0 Portfolio

# 🚀 AI-Powered Semantic Search Portfolio

An intelligent, modern web application that allows visitors to search through personal information using advanced AI semantic search technology. Built with cutting-edge web technologies and featuring a stunning glassmorphism UI design.

![Portfolio Demo](https://img.shields.io/badge/Status-Live-brightgreen)
![AI Powered](https://img.shields.io/badge/AI-Powered-blue)
![Modern UI](https://img.shields.io/badge/UI-Glassmorphism-purple)

## ✨ Features

### 🧠 **AI-Powered Search**
- **Semantic Understanding**: Uses transformer models to understand the meaning behind queries
- **Intelligent Matching**: Finds relevant information based on context, not just keywords
- **Real-time Processing**: Instant search results with modern ML models
- **Natural Language Queries**: Ask questions in plain English

### 🎨 **Modern UI/UX**
- **Glassmorphism Design**: Contemporary frosted glass effects with backdrop blur
- **Animated Gradients**: Dynamic, flowing background animations
- **Floating Particles**: Subtle interactive visual elements
- **Smooth Transitions**: Micro-animations throughout the interface
- **Responsive Design**: Perfect experience across all devices

### 🔧 **Technical Excellence**
- **Client-Side AI**: Runs entirely in the browser using Transformers.js
- **No Server Required**: Pure frontend implementation
- **Fast Loading**: Optimized for performance
- **Modern JavaScript**: ES6+ modules and async/await patterns

## 🛠️ Technology Stack

### **Frontend**
- **HTML5**: Semantic markup and modern web standards
- **CSS3**: Advanced styling with animations, gradients, and glassmorphism
- **Vanilla JavaScript**: ES6+ modules for clean, maintainable code

### **AI/ML**
- **Transformers.js**: Hugging Face transformers for the browser
- **GTE-Small Model**: General Text Embeddings for semantic search
- **Cosine Similarity**: Mathematical similarity computation
- **Feature Extraction Pipeline**: Text-to-vector conversion

### **Design**
- **Glassmorphism**: Modern design trend with frosted glass effects
- **Gradient Animations**: CSS keyframe animations
- **Responsive Grid**: Flexible layouts for all screen sizes
- **Inter Font**: Modern, readable typography

## 📁 Project Structure

```
portfolio-semantic-search/
├── index.html              # Main application file
├── bio.txt                 # Personal information database
├── README.md              # Project documentation
└── assets/                # (Optional) Additional resources
    ├── images/
    └── fonts/
```

## 🚀 Quick Start

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/ai-portfolio-search.git
cd ai-portfolio-search
```

### **2. Prepare Your Content**
Create a `bio.txt` file with your personal information, using `##` as section separators:

```text
## About Me
I'm a passionate developer with expertise in AI and web technologies...

## Skills
- JavaScript/TypeScript
- Python
- Machine Learning
- Web Development

## Experience
Software Engineer at TechCorp (2020-2023)
- Built scalable web applications
- Implemented AI-powered features
```

### **3. Serve the Application**
Since the app uses ES6 modules, you need to serve it through a web server:

#### **Option A: Using Python**
```bash
# Python 3
python -m http.server 3000

# Python 2
python -m SimpleHTTPServer 3000
```

#### **Option B: Using Node.js**
```bash
npx serve .
```

#### **Option C: Using VS Code**
Install the "Live Server" extension and right-click on `index.html` → "Open with Live Server"

### **4. Open in Browser**
Navigate to `http://localhost:3000` and start searching!

## 📊 How It Works

### **1. Content Processing**
- Loads biographical data from `bio.txt`
- Splits content into chunks using `##` delimiters
- Each chunk represents a searchable section

### **2. AI Processing**
- Converts user queries into numerical embeddings
- Transforms each content chunk into embeddings
- Uses the GTE-Small model for semantic understanding

### **3. Similarity Matching**
- Computes cosine similarity between query and content embeddings
- Finds the most semantically similar content chunk
- Returns the best matching information

### **4. User Interface**
- Modern, responsive design with glassmorphism effects
- Real-time search with loading animations
- Smooth transitions and micro-interactions

## 🎯 Usage Examples

### **Sample Queries**
- "What are your technical skills?"
- "Tell me about your work experience"
- "What projects have you worked on?"
- "What is your educational background?"
- "What programming languages do you know?"

### **Content Structure**
```text
## Technical Skills
JavaScript, Python, React, Node.js, Machine Learning, AI

## Professional Experience
Senior Developer at InnovateX
- Led development of AI-powered analytics platform
- Managed team of 5 developers
- Implemented microservices architecture

## Education
Master's in Computer Science
University of Technology (2018-2020)
```

## 🔧 Customization

### **Styling**
Modify the CSS variables in `index.html` to match your brand:

```css
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --accent-color: #a855f7;
    --text-color: #ffffff;
}
```

### **AI Model**
Change the transformer model by modifying the pipeline initialization:

```javascript
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
```

### **Content Format**
Customize the chunk delimiter by changing the split character:

```javascript
const chunks = text.split('---').map(chunk => chunk.trim());
```

## 📱 Browser Support

- **Chrome**: 88+ ✅
- **Firefox**: 87+ ✅
- **Safari**: 14+ ✅
- **Edge**: 88+ ✅

*Note: Requires modern browsers with ES6 module support*

## 🔒 Privacy & Security

- **Client-Side Processing**: All AI processing happens in the browser
- **No Data Transmission**: Personal information never leaves the user's device
- **No External APIs**: Completely self-contained application
- **No Tracking**: No analytics or tracking scripts

## 🚀 Performance

- **Model Size**: ~25MB (cached after first load)
- **Loading Time**: 2-5 seconds on first visit
- **Search Speed**: < 1 second for most queries
- **Memory Usage**: ~100MB peak during processing

## 🔄 Updates & Maintenance

### **Adding New Content**
1. Update `bio.txt` with new sections
2. Use `##` to separate different topics
3. Refresh the page to reload content

### **Updating the UI**
- Modify CSS in the `<style>` section
- Add new animations or effects
- Customize colors and typography


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For the amazing Transformers.js library
- **Xenova**: For the browser-optimized ML models
- **Design Community**: For glassmorphism and modern UI inspiration

## 📞 Contact & Support

- **Portfolio**: [Sumit Yadav](https://llm.sumityadav.com.np)
- **Email**: rockerritesh@gmail.com
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/rockerritesh)
- **GitHub**: [Your GitHub](https://github.com/rockerritesh)

---

**Made with ❤️ and AI by Sumit Yadav**

*Showcasing the future of interactive portfolios with semantic search technology*
