# Academic Website Development Checklist

## 📋 Phase 1: Foundation Setup (Week 1-2)

### Environment Setup
- [x] Install Ruby, Bundler, and Jekyll dependencies
- [x] Clone repository and test local build
- [x] Verify site loads at `http://localhost:4000` (using `bundle exec jekyll serve`)
- [x] Test hot-reload functionality for development

### Repository Configuration
- [ ] Set up GitHub repository permissions
- [ ] Configure GitHub Pages in repository settings
- [ ] Test deployment pipeline
- [ ] Set up branch protection rules (optional)

### Basic Site Configuration
- [ ] Update `_config.yml` with personal information
  - [x] Site title and description
  - [x] Personal name (first, middle, last)
  - [x] Contact information and preferred contact method
  - [ ] Site keywords for SEO
- [ ] Configure `_data/socials.yml`
  - [x] Email address
  - [x] GitHub username
  - [x] LinkedIn profile
  - [ ] ORCID ID
  - [x] Google Scholar profile
  - [x] ResearchGate profile (if applicable)
  - [x] X(Twitter) handle

---

## 📄 Phase 2: Core Content Pages (Week 2-4)

### About Page (`_pages/about.md`)
- [X] Update personal biography
- [x] Add research interests and current position
- [x] Upload professional photo (`assets/img/prof_pic.jpg`)
- [x] Configure profile image settings (circular/rectangular)
- [x] Update contact information section
- [x] Test social media links
- [x] Enable/disable news announcements
- [x] Configure selected papers display

### Publications Page (`_pages/publications.md`)
- [x] Enable publications page in navigation (`nav: true`)
- [x] Set up bibliography file (`_bibliography/papers.bib`)
  - [x] Clear fake/template publications data
  - [x] Add real publications in BibTeX format (5 real publications added)
  - [x] Include abstracts for key publications
  - [ ] Add PDF links where available
  - [x] Set up DOI links and HTML links
- [x] Configure author highlighting in `_config.yml`
  - [x] Update `scholar.last_name` and `scholar.first_name`
- [x] Set up co-author information (`_data/coauthors.yml`) with real collaborators
- [ ] Test bibliography search functionality
- [ ] Configure publication badges (Altmetric, citations)
- [x] Set up publication sorting (by year, importance)

### CV Page (`_pages/cv.md`)
- [x] Enable CV page in navigation (`nav: true`)
- [x] Remove description text from CV page
- [x] Fix text overlapping issues in Experience section
- [x] Handle PDF download button (commented out until PDF is available)
- [x] Fix YAML formatting issues in CV data
- [x] Add responsive design fixes for mobile devices
- [ ] Choose CV format (JSON or YAML) - Currently using YAML
- [x] Update `_data/cv.yml` with:
  - [x] Education history
  - [x] Work experience
  - [x] Skills and competencies
  - [ ] Publications (currently commented out)
  - [x] Awards and honors
  - [ ] Languages (currently commented out)
  - [ ] References (currently commented out)
- [ ] Add downloadable PDF version
- [ ] Configure CV sections visibility
- [ ] Test PDF download functionality
- [x] Set up table of contents sidebar

### Projects Page (`_pages/projects.md`)
- [x] Enable projects page in navigation (`nav: true`)
- [x] Plan project categories (research, work, fun)
- [x] Create individual project files in `_projects/`
  - [x] Project 1: Speech Entrainment Detection (PhD research)
  - [x] Project 2: Prosody Analysis Framework (current research)
  - [x] Project 3: Voice Similarity Analyzer (DAVI internship)
  - [x] Project 4: AI Teacher Response System (BEA shared task)
  - [ ] Project 5: [Add more as needed]
- [x] For each project file:
  - [x] Add project title and description
  - [x] Include project abstracts and technical details
  - [ ] Add links to code repositories (placeholder links added)
  - [x] Set importance ranking
  - [x] Assign to appropriate category
  - [x] Link related publications
- [ ] Configure project display layout (grid/horizontal)
- [ ] Test project filtering by category
- [ ] Add project preview images

---

## 📝 Phase 3: Content Expansion (Week 4-6)

### Blog Setup
- [x] Enable blog functionality in `_config.yml`
- [x] Configure pagination settings
- [x] Set up blog categories and tags (research, tutorials, thoughts)
- [x] Create first blog posts in `_posts/`
  - [x] Welcome post introducing yourself and research
  - [x] Technical tutorial post (Deep Learning for Speech Entrainment)
  - [x] Research reflection post (PhD to Postdoc journey)
- [ ] Set up comment system (Giscus)
  - [ ] Create GitHub discussion repository
  - [ ] Configure Giscus settings
  - [ ] Test commenting functionality
- [x] Configure related posts feature
- [x] Update blog name and description to be research-focused

### Teaching Page (`_pages/teaching.md`)
- [x] Enable teaching page in navigation (`nav: true`)
- [x] Update teaching philosophy and approach
- [x] Add current and past teaching experience
  - [x] University of Sheffield (2017-2019)
  - [x] University of Ljubljana (2015-2017)
- [x] Include professional development information
- [x] Add educational technology and innovation section
- [x] Document student mentoring and supervision experience
- [x] Outline future teaching interests
- [ ] Add links to course materials (if public)
- [ ] Consider creating a courses collection

### News Section
- [ ] Create news items in `_news/`
  - [ ] Recent research achievements
  - [ ] Conference presentations
  - [ ] Awards and recognitions
- [ ] Configure news display on homepage
- [ ] Set up news pagination
- [ ] Test news archive functionality

---

## 🚀 Phase 4: Advanced Features (Week 6-8)

### SEO and Performance
- [ ] Configure meta descriptions for all pages
- [ ] Set up Open Graph tags
- [ ] Add Schema.org structured data
- [ ] Set up Google Search Console
  - [ ] Verify site ownership
  - [ ] Submit sitemap
  - [ ] Monitor indexing status
- [ ] Enable image optimization (ImageMagick)
- [ ] Configure CSS/JS minification
- [ ] Test site loading speeds

### Analytics Integration
- [ ] Set up Google Analytics 4
  - [ ] Create GA4 property
  - [ ] Add tracking code to `_config.yml`
  - [ ] Configure goals and events
  - [ ] Set up conversion tracking
- [ ] Configure academic metrics
  - [ ] Enable Altmetric badges
  - [ ] Set up Google Scholar citation tracking
  - [ ] Configure publication metrics display

### Interactive Features
- [ ] Enable site-wide search functionality
- [ ] Configure search for posts and publications
- [ ] Test search performance and relevance
- [ ] Set up search analytics
- [ ] Configure social media sharing buttons
- [ ] Test responsive design on mobile devices

---

## 📊 Phase 5: Monitoring and Analytics (Week 8-10)

### Visitor Analytics Setup
- [ ] Create comprehensive analytics dashboard
- [ ] Set up custom Google Analytics reports
  - [ ] Page popularity tracking
  - [ ] User engagement metrics
  - [ ] Traffic source analysis
  - [ ] Academic content performance
- [ ] Configure goal tracking for:
  - [ ] Publication downloads
  - [ ] Contact form submissions
  - [ ] Social media clicks
  - [ ] Newsletter signups (if enabled)

### Academic Impact Tracking
- [ ] Monitor publication page views
- [ ] Track PDF download statistics
- [ ] Set up referral tracking from academic databases
- [ ] Monitor citations and mentions
- [ ] Create monthly analytics reports

### Performance Monitoring
- [ ] Set up Lighthouse CI for performance monitoring
- [ ] Configure uptime monitoring service
- [ ] Monitor site loading speeds across devices
- [ ] Set up broken link monitoring
- [ ] Monitor search engine rankings

---

## 🔄 Phase 6: Maintenance and Growth (Ongoing)

### Content Management
- [ ] Establish content update schedule
  - [ ] Weekly blog posts (if maintaining blog)
  - [ ] Monthly publications updates
  - [ ] Quarterly project updates
  - [ ] Annual CV updates
- [ ] Create content calendar
- [ ] Set up backup procedures
- [ ] Monitor template updates

### Community Engagement
- [ ] Share new content on social media
- [ ] Engage with academic Twitter/LinkedIn
- [ ] Participate in academic discussions
- [ ] Submit site to academic directories
- [ ] Network with other researchers using al-folio

### Technical Maintenance
- [ ] Regular dependency updates
- [ ] Security monitoring
- [ ] Performance optimization
- [ ] Backup verification
- [ ] Monitor for broken links

---

## 🎯 Success Checkpoints

### Phase 1 Complete ✅
- [ ] Site builds locally without errors
- [ ] Basic information displays correctly
- [ ] Social media links work

### Phase 2 Complete ✅
- [ ] All four core pages have real content
- [ ] Publications display correctly with proper formatting
- [ ] CV sections are populated and PDF downloads
- [ ] Projects showcase your work effectively

### Phase 3 Complete ✅
- [ ] Blog is functional with at least 3 posts
- [ ] Teaching page reflects your experience
- [ ] News section displays recent updates

### Phase 4 Complete ✅
- [ ] Google Analytics tracking works
- [ ] SEO optimizations implemented
- [ ] Site search functions properly
- [ ] Academic metrics display correctly

### Phase 5 Complete ✅
- [ ] Analytics dashboard provides meaningful insights
- [ ] Performance monitoring alerts work
- [ ] Academic impact metrics are tracked

### Phase 6 Ongoing ✅
- [ ] Content updated regularly
- [ ] Community engagement active
- [ ] Technical health maintained

---

## 📞 Support Resources

- **Template Documentation**: [al-folio CUSTOMIZE.md](../CUSTOMIZE.md)
- **Jekyll Documentation**: https://jekyllrb.com/docs/
- **Template FAQ**: [al-folio FAQ.md](../FAQ.md)
- **Community Support**: al-folio GitHub discussions
- **Your Development Partner**: Ready to help debug and implement features!

---

## 📝 Notes Section

Use this space to track your progress, note customizations, and record decisions:

```
Date: _______
Progress Notes:
________________
________________
________________

Custom Modifications Made:
________________
________________
________________

Next Steps:
________________
________________
________________
```
