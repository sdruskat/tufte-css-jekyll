---
layout: page
title: Tufte CSS
subtitle: Dave Liepmann

---

Tufte CSS provides tools to style web articles using the ideas demonstrated by Edward Tufte’s books and handouts. Tufte’s style is known for its simplicity, extensive use of sidenotes, tight integration of graphics with text, and carefully chosen typography.

Tufte CSS was created by Dave Liepmann and is now an Edward Tufte project. The original idea was cribbed from Tufte-{% latex %} and R Markdown’s Tufte Handout format. We give hearty thanks to all the people who have contributed to those projects.

If you see anything that Tufte CSS could improve, we welcome your contribution in the form of an issue or pull request on the GitHub project: tufte-css. Please note the contribution guidelines.

Finally, a reminder about the goal of this project. The web is not print. Webpages are not books. Therefore, the goal of Tufte CSS is not to say “websites should look like this interpretation of Tufte’s books” but rather “here are some techniques Tufte developed that we’ve found useful in print; maybe you can find a way to make them useful on the web”. Tufte CSS is merely a sketch of one way to implement this particular set of ideas. It should be a [starting point](http://google.de), not a design goal, because any project should present their information as best suits their particular circumstances.



{% newthought 'In this first iteration chumba'%} of the *Tufte-Jekyll* theme, a post and a page have exactly the same layout. That means that all the typographic and structural details are identical between the two.

## Pages and Posts

CHUG CHUG Jekyll provides for both pages and posts in its default configuration. I have left this as-is. 

### Posts

Conceptually, posts are for recurring pieces of similar content such as might be found in a typical blog entry. Post content all sits in a folder named ```_posts``` and files are created in this folder{% sidenote 1 'Yes, a page has essentially the same old shit as a post'%} that are names with a date pre-pended to the title of the post. For instance ```2105-02-20-this-is-a-post.md``` is a perfectly valid post filename.

Posts will always have a section above the content itself consisting of YAML front matter, which is meta-data information about the post. Minimally, a post title must always be present for it to be processed properly.

```
---
Title: Some Title
---
## Content

Markdown formatted content *here*.
```


### Pages

Pages are any HTML documents *or* Markdown documents with YAML front matter that are then converted to content. Page material is more suited to static, non-recurring types of content. Like this

I am not going to re-write the Jekyll documentation. Read it and you will figure out how the site is structured.

