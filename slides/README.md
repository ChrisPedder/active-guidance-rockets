# Rocket Stabilisation Seminar Slides

Interactive slides for a seminar at the CS/ML lecture series at HES-SO, built with [Reveal.js](https://revealjs.com/).

## 📁 Structure

```
slides/
├── index.html         # Main presentation file
├── slides.md          # Markdown to define slides
├── images/            # (Optional) Directory for images
├── videos/            # (Optional) Directory for videos
└── README.md          # This file
```

## 🚀 Viewing the Slides

### Online (GitHub Pages)
Once you've enabled GitHub Pages for your repository:
- **URL:** `https://[your-username].github.io/active_guidance_rockets/slides/`

### Locally
1. Simply open `index.html` in a web browser
2. Or use a local server (recommended for full functionality):
   ```bash
   # Using Python 3
   cd slides
   uv run python -m http.server 8000
   # Then visit: http://localhost:8000
   ```

## 🎮 Navigation & Controls

- **Arrow Keys:** Navigate slides
  - Right/Left: Main sections
  - Up/Down: Sub-sections within a topic
- **Space:** Next slide
- **ESC:** Overview mode (see all slides)
- **S:** Speaker notes (opens separate window)
- **F:** Fullscreen
- **?:** Show all keyboard shortcuts

## 📝 Editing the Slides

### Quick Edits (GitHub Web Interface)
1. Click on `slides.md`
2. Click the pencil icon (Edit)
3. Make your changes using Markdown
4. Commit changes
5. Slides update automatically!

### Local Editing
1. Edit `slides.md` in any text editor
2. Save and refresh your browser
3. Commit and push to GitHub

### Markdown Syntax

#### Slide Separators
```markdown
---           # New main slide (horizontal)
----          # New sub-slide (vertical)
Note:         # Speaker notes (only visible in notes window)
```

#### Basic Formatting
```markdown
# Title
## Subtitle
### Heading

**bold text**
*italic text*
`code`

- Bullet point
1. Numbered list

[Link text](URL)
![Image alt text](image-url.jpg)
```

#### Math Equations
```markdown
Inline: $E = mc^2$

Block:
$$
\frac{-b \pm \sqrt{b^2-4ac}}{2a}
$$
```

#### Code Blocks
````markdown
```python
def hello_world():
    print("Hello, AI students!")
```
````

#### Special Classes
```markdown
<span class="key-term">Important Term</span>
<span class="emphasis">Emphasized Text</span>
<span class="small">Small note or citation</span>
```

#### Two-Column Layout
```markdown
<div class="columns">
<div class="column">

Content for left column

</div>
<div class="column">

Content for right column

</div>
</div>
```

## 🎨 Customization

### Change Theme
Edit `index.html` and change the theme link:
```html
<!-- Available themes: black, white, league, beige, sky, night, serif, simple, solarized -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4.5.0/dist/theme/white.css">
```

### Modify Styling
Custom CSS is in `<style>` tags in `index.html`. Key variables:
- Font sizes
- Colors for headings
- Spacing
- Custom classes

### Add Images
1. Create `images/` directory in `slides/`
2. Add your images
3. Reference in markdown: `![Description](images/your-image.png)`

## 📤 Exporting

### PDF Export
1. Add `?print-pdf` to the URL
   - Example: `http://localhost:8000/?print-pdf`
2. Open print dialog (Ctrl/Cmd + P)
3. Select "Save as PDF"
4. Adjust settings (remove headers/footers)
5. Save

Perfect for:
- Distribution to students
- Archival purposes

## 🔧 Advanced Features

### Adding More Parts
To add `part_c.md`:
1. Create the markdown file
2. Edit `index.html` and add:
```html
<section data-markdown="part_c.md"
         data-separator="^\n---\n$"
         data-separator-vertical="^\n----\n$"
         data-separator-notes="^Note:">
</section>
```

### Fragments (Appear Incrementally)
```markdown
- First point <!-- .element: class="fragment" -->
- Second point <!-- .element: class="fragment" -->
- Third point <!-- .element: class="fragment" -->
```

### Auto-Animate
```markdown
---
<!-- .slide: data-auto-animate -->
## Step 1

---
<!-- .slide: data-auto-animate -->
## Step 2
More content appears smoothly
```

## 🛠️ Troubleshooting

**Slides not loading:**
- Check browser console for errors (F12)
- Ensure you're using a local server (not just `file://`)
- Verify markdown syntax in `.md` files

**Math not rendering:**
- Clear browser cache
- Check that equations use `$` or `$$` correctly

**Images not showing:**
- Verify image paths are correct
- Use relative paths: `images/myimage.png`
- Ensure images are committed to git

## 📚 Resources

- [Reveal.js Documentation](https://revealjs.com/)
- [Markdown Guide](https://www.markdownguide.org/)
- [KaTeX Math Syntax](https://katex.org/docs/supported.html)
